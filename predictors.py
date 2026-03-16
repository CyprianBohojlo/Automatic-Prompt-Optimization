from __future__ import annotations
import concurrent.futures, os
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import utils                      # original OpenAI wrapper
import vectorize                  # builds the Chroma retriever
import threading


class GPT4Predictor(ABC):
    # Minimal interface every predictor must implement
    def __init__(self, opt: Dict | None = None):
        self.opt = opt or {}

    @abstractmethod
    def inference(self, ex: Dict, prompt: str) -> str:
        pass

    def batch_inference(self, examples: List[Dict], prompt: str) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self.inference, ex, prompt) for ex in examples
            ]
            return [f.result() for f in futures]


# Binary Predictor kept just in case
class BinaryPredictor(GPT4Predictor):
    categories = {0: "No", 1: "Yes"}

    def __init__(self, opt: Dict | None = None):
        super().__init__(opt)
        self.temperature = self.opt.get("temperature", 0.0)

    def _render(self, ex: Dict, prompt: str) -> str:
        return prompt.format(text=ex["text"])

    def inference(self, ex: Dict, prompt: str) -> str:
        rendered = self._render(ex, prompt)
        answer   = utils.chatgpt(
            rendered,
            temperature=self.temperature,
            max_tokens=self.opt.get("max_tokens", 1),
            n=1,
            timeout=10,
        )[0]
        answer = answer.lower()
        if "yes" in answer:
            return "Yes"
        if "no" in answer:
            return "No"
        # default fallback
        return answer.strip()


class QA_Generator(GPT4Predictor):
    def __init__(self, opt=None):
        super().__init__(opt)
        self.top_k = self.opt.get("top_k", 3)
        self.rag_method = self.opt.get("rag_method", "neural")
        self.retrievers = {}           # doc_name → retriever (neural)
        self.lock = threading.Lock()

        # Load Sturdy bulk index + doc_id mapping if needed
        if self.rag_method == "sturdy":
            mapping_path = self.opt.get("doc_id_mapping")
            if not mapping_path:
                raise ValueError("--doc_id_mapping is required when --rag_method=sturdy")

            self._doc_id_lookup = self._build_doc_id_lookup(mapping_path)

            from sturdystats.index import Index as SturdyIndex
            index_name = self.opt.get("sturdy_index_name", "bulk_train_all")
            self._sturdy_api_key = os.environ["STURDY_API_KEY"]
            self._sturdy_index = SturdyIndex(
                name=index_name, API_key=self._sturdy_api_key
            )

    @staticmethod
    def _build_doc_id_lookup(mapping_path: str) -> Dict[str, str]:
        """Build a lookup dict from various document identifiers to sturdy_doc_id.

        The mapping CSV has different identifier columns per dataset:
          - FinanceBench: doc_name  (e.g. "3M_2018_10K")
          - FinDoc-RAG:   filename  (e.g. "873f.md")
          - DocFinQA:     doc_label (e.g. "DocFinQA_0")

        We index by all available identifiers so the pipeline can look up
        by whatever doc_name field appears in the example dict.
        """
        df = pd.read_csv(mapping_path)
        lookup = {}
        for _, row in df.iterrows():
            doc_id = row.get("sturdy_doc_id")
            if pd.isna(doc_id):
                continue
            # Register every non-null identifier
            for col in ("doc_name", "filename", "doc_label"):
                val = row.get(col)
                if pd.notna(val):
                    lookup[str(val)] = doc_id
                    # Also register filename without .md extension
                    if col == "filename" and str(val).endswith(".md"):
                        lookup[str(val).removesuffix(".md")] = doc_id
        return lookup

    # ---- neural (Chroma / e5) retrieval ----
    def get_retriever(self, doc_name):
        if doc_name in self.retrievers:          # fast path
            return self.retrievers[doc_name]
        with self.lock:                          # first builder wins
            if doc_name not in self.retrievers:  # 2nd check inside lock
                r = vectorize.build_vectorstore_retriever(doc_name)
                r.search_kwargs["k"] = self.top_k
                self.retrievers[doc_name] = r
        return self.retrievers[doc_name]

    def _retrieve_neural(self, doc_name, question):
        docs = self.get_retriever(doc_name).invoke(question)
        return "\n".join(d.page_content for d in docs)

    # ---- sturdy (hierarchical Bayesian LM) retrieval ----
    def _retrieve_sturdy(self, doc_name, question):
        sturdy_doc_id = self._doc_id_lookup.get(doc_name)
        if sturdy_doc_id is None:
            raise KeyError(
                f"No sturdy_doc_id found for doc '{doc_name}' in mapping"
            )
        results = self._sturdy_index.query(
            search_query=question,
            limit=self.top_k,
            context=1,
            filters=f"d.doc_id='{sturdy_doc_id}'",
            semantic_search_weight=0.8,
            semantic_search_cutoff=0.1,
            return_df=True,
        )
        return "\n".join(results["text"].tolist())

    # ---- unified inference ----
    def inference(self, ex, prompt):
        if "{question}" not in prompt or "{context}" not in prompt:
            raise KeyError("Prompt must contain {question} and {context} placeholders")
        if self.rag_method == "sturdy":
            ctx = self._retrieve_sturdy(ex["doc_name"], ex["question"])
        else:
            ctx = self._retrieve_neural(ex["doc_name"], ex["question"])
        filled_prompt = prompt.format(question=ex["question"], context=ctx)
        answer = utils.chatgpt(filled_prompt, temperature=0.0, n=1, timeout=15)[0]
        return answer.strip()
