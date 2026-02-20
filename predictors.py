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
        self.sturdy_indices = {}       # doc_name → Index    (sturdy)
        self.lock = threading.Lock()

        # Load Sturdy manifest if needed
        if self.rag_method == "sturdy":
            manifest_path = self.opt.get("sturdy_manifest")
            if not manifest_path:
                raise ValueError("--sturdy_manifest is required when --rag_method=sturdy")
            self._manifest = pd.read_csv(manifest_path)
            from sturdystats.index import Index as SturdyIndex
            self._SturdyIndex = SturdyIndex
            self._sturdy_api_key = os.environ["STURDY_API_KEY"]

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

    def _get_sturdy_index(self, doc_name):
        if doc_name in self.sturdy_indices:
            return self.sturdy_indices[doc_name]

        with self.lock:
            if doc_name not in self.sturdy_indices:
                row = self._manifest[self._manifest["doc_name"] == doc_name]
                if row.empty:
                    raise KeyError(f"No Sturdy index found for doc '{doc_name}' in manifest")
                index_name = row.iloc[0]["index_name"]
                idx = self._SturdyIndex(name=index_name, API_key=self._sturdy_api_key)
                self.sturdy_indices[doc_name] = idx
        return self.sturdy_indices[doc_name]

    def _retrieve_sturdy(self, doc_name, question):
        idx = self._get_sturdy_index(doc_name)
        results = idx.query(
            search_query=question,
            limit=self.top_k,
            context=1,
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


