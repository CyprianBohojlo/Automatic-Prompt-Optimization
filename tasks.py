from __future__ import annotations
import concurrent.futures, json, os, pathlib, random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
import random

from scorers import BEMScorer    


@dataclass
class InputExample:
    guid: str
    question: str
    answer: str


class DataProcessor(ABC):
    def __init__(self, data_dir: str, max_threads: int = 4):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self) -> List[InputExample]:
        pass

    @abstractmethod
    def get_test_examples(self) -> List[InputExample]:
        pass

    @abstractmethod
    def evaluate(self, predictor, prompt, test_exs, n: int | None = None):
        pass

    # retained only for API compatibility
    def stringify_prediction(self, pred):  
        return pred.strip()


def process_example(ex: Dict, predictor, prompt):
    """Called in worker, returns both example and prediction."""
    pred = predictor.inference(ex, prompt)
    return ex, pred


class FinanceBenchTask(DataProcessor):
    SPLIT_FILE = "financebench_split.json"

    def __init__(self, data_dir: str, max_threads: int = 4, seed: int = 42):
        super().__init__(data_dir, max_threads)
        self.seed = seed
        self.df: pd.DataFrame = pd.read_parquet(
            pathlib.Path(data_dir) / "dataset_prepared.parquet"
        )
        self.train_idx, self.test_idx = self._load_or_make_split()

        # Random shuffle, can be removed
        random.Random(self.seed).shuffle(self.train_idx)

        self.bem_scorer = BEMScorer(None, bem_threshold=0.56)

    # Making a split 80/20 if it's not already loaded
    def _load_or_make_split(self):
        path = pathlib.Path(self.data_dir) / self.SPLIT_FILE
        if path.exists():
            with open(path, "r") as f:
                split = json.load(f)
            return split["train"], split["test"]

        idx = list(range(len(self.df)))
        random.Random(self.seed).shuffle(idx)
        k = int(0.8 * len(idx))
        split = {"train": idx[:k], "test": idx[k:]}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(split, f)
        return split["train"], split["test"]

    
    def _idx_to_examples(self, idxs):
        sub = self.df.iloc[idxs]
        return [
            {
                "id":       int(i),
                "question": r.question,
                "answer":   r.answer,
                "doc_name": r.doc_name,        
            }
            for i, r in sub.iterrows()
        ]

    def get_train_examples(self):
        return self._idx_to_examples(self.train_idx)

    def get_test_examples(self):
        return self._idx_to_examples(self.test_idx)
    

    # Evaluation using the BEM method
    def run_evaluate(self, predictor, prompt, examples: List[Dict], n: int | None = None) -> float:
        exs = examples if n is None else examples[:n]

        texts = []
        labels = []
        preds = []       
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as pool:
            futures = [pool.submit(process_example, ex, predictor, prompt) for ex in exs]

            for ex, pred in tqdm((f.result() for f in futures), total=len(futures), desc="BEM eval"):
                texts.append(ex["question"])
                labels.append(ex["answer"])
                preds.append(pred)

        # BEM scoring loop 
        hits = sum(
            self.bem_scorer.pair_equivalent(p, g, q)
            for p, g, q in zip(preds, labels, texts)
        )
        return hits / len(exs)
    
    def evaluate(self, predictor, prompt, examples, n=None):
        return self.run_evaluate(predictor, prompt, examples, n)


# Later used in main.py
def get_task(task_name: str, data_root: str, **kw) -> FinanceBenchTask:
    if task_name.lower() != "financebench":
        raise ValueError("Only FinanceBench is supported in this fork.")
    return FinanceBenchTask(data_root, **kw)
