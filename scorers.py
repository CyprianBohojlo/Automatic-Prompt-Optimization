from __future__ import annotations
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from transformers import AutoTokenizer

BEM_URL: str = ('https://kaggle.com/models/google/bert/frameworks/TensorFlow2/variations/answer-equivalence-bem/versions/1')

BEM_MODEL = hub.load(BEM_URL)
_BEM_LOCK  = threading.Lock()
BEM_tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
BEM_threshold   = 0.56



def predict_on_example(inputs):
    """
    Worker helper for concurrent evaluation.
    Returns: (prompt_text, example_dict, LLM_prediction_string)
    """
    ex, predictor, prompt = inputs
    pred = predictor.inference(ex, prompt)
    return prompt, ex, pred



class BaseScorer(ABC):
    """
    Implements a prompt-hash and example-id, prediction cache so the
    optimiser never calls the LLM twice for the same pair.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self._pred_cache: Dict[Tuple[str, int], str] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        #return hashlib.md5(prompt.encode()).hexdigest()[:12]
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    # scorers.py  –  replace the whole _predict method
    def _predict(self, examples: List[Dict], prompt_text: str) -> List[str]:
        ph = self._hash_prompt(prompt_text)
        missing = []
        # 1) identify which examples still need a call
        for ex in examples:
            key = (ph, ex["id"])
            if key not in self._pred_cache:
                missing.append(ex)

        # 2) fetch only the missing ones
        if missing:
            new_preds = self.predictor.batch_inference(missing, prompt_text)
            with self._cache_lock:
                for ex, p in zip(missing, new_preds):
                    self._pred_cache[(ph, ex["id"])] = p

        # 3) now build the prediction list in the original order
        return [self._pred_cache[(ph, ex["id"])] for ex in examples]


    @abstractmethod
    def __call__(self, examples: List[Dict], prompt_text: str) -> float:
        ...



class BEMScorer(BaseScorer):
    
    # pair_prob returns float in [0,1]
    # pair_equivalent returnsboolean using self.tau
    # __call__  returns mean boolean accuracy over the minibatch
    

    def __init__(self, predictor, bem_threshold: float = BEM_threshold):
        super().__init__(predictor)
        self.tau = bem_threshold
        global BEM_MODEL
        

    def pair_prob(self, pred: str, gold: str, question: str) -> float:
        encoding = BEM_tokenizer(pred, gold, truncation=True, max_length=512, padding="max_length", return_tensors="tf")
        inputs = {
            "input_ids":tf.cast(encoding["input_ids"], tf.int64),
            "segment_ids": tf.cast(encoding["token_type_ids"], tf.int64),
        }
        with _BEM_LOCK:
            logits = BEM_MODEL(inputs)
             # If the SavedModel ever returns a dict on another platform, keep the following safeguard:
        if isinstance(logits, dict):           
            logits = list(logits.values())[0]
        prob_equiv = tf.nn.softmax(logits, axis=-1).numpy().squeeze()[1]
        return float(prob_equiv)

    def pair_equivalent(self, pred: str, gold: str, question: str) -> bool:
        return self.pair_prob(pred, gold, question) >= self.tau

    def __call__(self, examples: List[Dict], prompt_text: str) -> float:
        preds = self._predict(examples, prompt_text)
        hits = [
            self.pair_equivalent(p, ex["answer"], ex["question"])
            for p, ex in zip(preds, examples)
        ]
        return float(np.mean(hits))
