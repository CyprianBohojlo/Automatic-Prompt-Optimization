import urllib3
import math
import numpy as np
import random
from tqdm import tqdm
import concurrent.futures
import requests
from PPO import PPO, Memory
from DPO import DPO
import numpy as np
import random
import torch


def _score_prompts(scorer, prompts, examples):
    
    #Returns a list[float] where each element is BEM scorer(examples, prompt).
    
    return [scorer(examples, p) for p in prompts]



class SuccessiveHalvingEvaluator:
    # Successive Halving Evaluator taken from original

    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10, samples_per_eval=5, max_threads=1,
        verbose=False, budget=None):

        out_ranks = [-1] * len(prompts)
        prompt2idx = {p: i for i, p in enumerate(prompts)}
        idx2prompts = {i: p for i, p in enumerate(prompts)}

        if budget is None:
            budget = self.config["eval_budget"]
        n = len(prompts)
        S = prompts[:]  # surviving prompts 

        for r in range(0, math.ceil(math.log2(n))):
            t_r = math.floor(budget / (len(S) * math.ceil(math.log2(n))))
            sample = random.sample(exs, min(len(exs), t_r))

            scores = _score_prompts(scorer, S, sample) # new adjusted to scorer
            average = np.mean(scores)

            for score, prompt in zip(scores, S):
                if score < average:
                    out_ranks[prompt2idx[prompt]] = r

            S = [p for p, s in zip(S, scores) if s >= average]

        # fill beam with default rank if not assigned
        r += 1
        for i in range(len(out_ranks)):
            if out_ranks[i] == -1:
                out_ranks[i] = r

        return out_ranks



class SuccessiveRejectsEvaluator:
     # Successive Rejects Evaluator taken from original

    def __init__(self, config):
        self.config = config

    def __call__( self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10, samples_per_eval=5,
        max_threads=1, verbose=False):
        assert self.config["evaluator"] in {"sr", "s-sr"}

        out_ranks = [-1] * len(prompts)
        idx2prompt = {i: p for i, p in enumerate(prompts)}
        num_rounds = len(prompts) - self.config["beam_size"]

        if self.config["evaluator"] == "s-sr":
            samples_per_round = math.ceil(self.config["eval_budget"] / (num_rounds * num_prompts_per_round))

            if samples_per_round == 0:
                raise ValueError("Eval budget too small for s-sr.")
            
        else:  # plain SR
            K = len(prompts) - self.config["beam_size"]
            log_bar_K = 0.5 + sum(1.0 / i for i in range(2, K + 1))
            n_prev_k = 0

        ri = 1
        with tqdm(total=len(idx2prompt), desc="sr") as pbar:
            while len(idx2prompt) > self.config["beam_size"]:

                if self.config["evaluator"] == "s-sr":
                    selected_data = random.sample(exs, samples_per_round)
                    sel_idx, sel_prompts = zip(
                        *random.sample(
                            idx2prompt.items(),
                            min(num_prompts_per_round, len(idx2prompt)),
                        )
                    )
                else:  # plain SR
                    sel_idx, sel_prompts = zip(*idx2prompt.items())
                    n_k = (1.0 / log_bar_K) * (
                        (self.config["eval_budget"] - K) / (K + 1 - ri)
                    )
                    samples_per_round = max(4, int(n_k - n_prev_k))
                    selected_data = random.sample(
                        exs, min(len(exs), samples_per_round)
                    )
                    n_prev_k = n_k

                scores = _score_prompts(scorer, sel_prompts, selected_data)
                min_idx = scores.index(min(scores))
                idx_to_remove = sel_idx[min_idx]

                del idx2prompt[idx_to_remove]
                out_ranks[idx_to_remove] = ri
                ri += 1
                pbar.update(1)

        # fill surviving prompts with final rank
        ri += 1
        for i in range(len(out_ranks)):
            if out_ranks[i] == -1:
                out_ranks[i] = ri
        return out_ranks


class UCBBandits:
    #Upper Confidence Bound algorithm taken from original

    def __init__(self, num_prompts, num_samples=5, c=1.0, mode="ucb"):
        self.c = c
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, s in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += s * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        return np.divide( self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, t):
        if np.sum(self.counts) == 0:
            return random.sample(range(self.num_prompts), n)
        scores = self.get_scores()
        counts = self.counts + 1e-3
        if self.mode == "ucb":
            ucb_scores = scores + self.c * np.sqrt(np.log(t) / counts)
        else:  # ucb-e
            ucb_scores = scores + self.c * np.sqrt(self.c / counts)
        return np.argsort(ucb_scores)[::-1][:n]

    def get_infos(self):
        return self.counts


class UCBBanditEvaluator:
    # Upper Confidence Bound Evaluator taken from original

    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10, samples_per_eval=5,
        max_threads=1, verbose=True):

        bandit_algo = UCBBandits(len(prompts), num_samples=samples_per_eval, mode=self.config["evaluator"], c=self.config["c"])

        for ri in tqdm(range(rounds), desc="UCB"):
            sampled_idx = bandit_algo.choose(min(num_prompts_per_round, len(prompts)), ri)
            sampled_prompts = [prompts[i] for i in sampled_idx]
            sampled_data = random.sample(exs, samples_per_eval)

            scores = _score_prompts(scorer, sampled_prompts, sampled_data)
            bandit_algo.update(sampled_idx, scores)

        return bandit_algo.get_scores().tolist()



class BruteForceEvaluator:
     # Brute Force Evaluator taken from original

    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40, num_prompts_per_round=10, c=2.0,
        samples_per_eval=5, max_threads=1, verbose=True):
        
        sample_size = min(len(exs), int(self.config["eval_budget"] / len(prompts)))
        eval_exs = random.sample(exs, sample_size)
        return _score_prompts(scorer, prompts, eval_exs)
    

class PPOEvaluator:
    # PPO evaluator used to score prompts 
    def __init__(self, eval_rounds: int = 50,
        samples_per_eval: int = 16,
        ppo_hidden: int = 64,
        ppo_lr: float = 2e-3,
        ppo_gamma: float = 0.99,
        log_history: bool = True):

        self.eval_rounds = eval_rounds
        self.samples_per_eval = samples_per_eval
        self.hidden = ppo_hidden
        self.lr = ppo_lr
        self.gamma = ppo_gamma
        self.log_history = log_history
        self.rewards_history = []
        self.rewards_full_history = []

    def reward(self, prompt, examples, scorer):
        return scorer(examples, prompt) # Returns mean accuracy
    
    def __call__(self, prompts, examples, task, predictor, scorer, random_number=None, **_):
        if random_number is None:
            random_number = random.Random()

        # create fresh agent sized to current prompt list
        ppo = PPO(action_dim=len(prompts), hidden=self.hidden, lr=self.lr, gamma=self.gamma)
        memory = Memory()

        train_pool = examples

        for _ in range(self.eval_rounds):
            idx = ppo.select_action(None, memory)
            batch = random_number.sample(train_pool, self.samples_per_eval)
            #rew = self.reward(prompts[idx], batch, scorer)  Old version without full hsitory 
            # --- compute per-example 0/1 hits, then the mean for training ---
            preds = scorer._predict(batch, prompts[idx])   # uses cache
            hits = [1.0 if scorer.pair_equivalent(p, ex["answer"], ex["question"]) else 0.0
                    for p, ex in zip(preds, batch)]
            rew = sum(hits) / len(hits)
            if self.log_history:
                self.rewards_history.append(rew) # Save rewards hisotry so we can plot it later
                self.rewards_full_history.append(hits)
            # store transition & update immediately 
            memory.rewards.append(rew)
            memory.is_terminals.append(True)

            ppo.update(memory)
            memory.clear_memory()

        # final preference scores the optimiser will rank by
        return ppo.get_action_preferences().tolist()
    

class DPOEvaluator:
    """
    Scores prompts with Direct-Preference-Optimisation.

    Expects:
      - scorer._predict(batch, prompt)  -> list[str]  (batched, cached LLM outputs)
      - scorer.pair_prob(pred, gold, q) -> float in [0,1]
      - (optional) scorer.batch_pair_prob(preds, golds, questions) -> list[float] (vectorized BEM)

    History (one entry per DPO update):
      self.history = {
        "loss": [],      # DPO preference loss
        "pairs": [],     # # of (w,l) pairs used
        "margin": [],    # mean winner-loser BEM gap
        "entropy": [],   # H(pi) after update
        "exp_prob": [],  # E_{a~pi}[BEM] on the active set
        "kl_ref": [],    # KL(pi || pi_ref) if anchored, else None
      }
    """

    def __init__(
        self,
        eval_rounds: int = 50,
        samples_per_eval: int = 16,
        dpo_hidden: int = 128,        # kept for CLI compat; not used by current logits-only policy
        dpo_lr: float = 3e-4,
        dpo_beta: float = 0.1,
        dpo_margin: float = 0.0,
        reference_free: bool = False,
        **_
    ):

        self.eval_rounds = eval_rounds
        self.samples_per_eval = samples_per_eval
        self.margin_tau = dpo_margin

        self.dpo = DPO(
            n_actions=1,                # lazily resized on first call
            beta=dpo_beta,
            lr=dpo_lr,
            hidden=dpo_hidden,          # ignored by logits-only Policy but kept for API compat
            reference_free=reference_free,
        )

        # logging buffers
        self.history = {
            "loss": [],
            "pairs": [],
            "margin": [],
            "entropy": [],
            "exp_prob": [],
            "kl_ref": [],
        }

    def _build_scores_matrix(self, scorer, batch, active_prompts, all_preds):
        """
        Returns scores_mat of shape (B, N_active), where entry (j,i) is
        BEM probability that all_preds[i][j] is equivalent to gold for example j.
        Uses scorer.batch_pair_prob if available; falls back to per-example pair_prob.
        """

        B = len(batch)
        N_active = len(active_prompts)

        golds = [ex["answer"] for ex in batch]
        qs    = [ex["question"] for ex in batch]

        # Fast path: vectorized BEM per prompt (one call per column)
        if hasattr(scorer, "batch_pair_prob"):
            cols = []
            for i in range(N_active):
                probs = scorer.batch_pair_prob(all_preds[i], golds, qs)  # len B
                cols.append(np.asarray(probs, dtype=np.float32))
            scores_mat = np.stack(cols, axis=1)  # (B, N_active)
            return scores_mat

        # Fallback: per-example calls
        scores_mat = np.empty((B, N_active), dtype=np.float32)
        for i in range(N_active):
            for j in range(B):
                scores_mat[j, i] = float(
                    scorer.pair_prob(all_preds[i][j], golds[j], qs[j])
                )
        return scores_mat

    def __call__(
        self,
        prompts,
        examples,
        task,
        predictor,
        scorer,
        num_prompts_per_round=None,
        **_,
    ):


        N_total = len(prompts)

        # Resize policy head if the number of actions changed
        if N_total != self.dpo.policy.logits.numel():
            from DPO import DPO
            self.dpo = DPO(
                n_actions=N_total,
                beta=self.dpo.beta,
                lr=self.dpo.optimizer.param_groups[0]["lr"],
                reference_free=self.dpo.reference_free,
            )

        for _ in range(self.eval_rounds):
            # Sample a mini-batch of examples
            batch = random.sample(examples, k=self.samples_per_eval)
            B = len(batch)

            # Sample an active subset of prompts for this iteration
            if (num_prompts_per_round is not None) and (num_prompts_per_round < N_total):
                active_idx = random.sample(range(N_total), k=num_prompts_per_round)
            else:
                active_idx = list(range(N_total))
            active_prompts = [prompts[i] for i in active_idx]
            N_active = len(active_prompts)

            # Batched, cached LLM predictions: one list per active prompt
            all_preds = [scorer._predict(batch, p) for p in active_prompts]  # len N_active, each len B

            # Build BEM score matrix once, reuse it
            scores_mat = self._build_scores_matrix(scorer, batch, active_prompts, all_preds)  # (B, N_active)

            # Winner/loser mining (hard negative) with margin filtering
            winners, losers, margins = [], [], []
            row_argmax = scores_mat.argmax(axis=1)  # best per example in ACTIVE set
            for j in range(B):
                w_local = int(row_argmax[j])
                row = scores_mat[j]

                if N_active < 2:
                    continue  # cannot form a pair with 1 prompt

                # pick the best loser (2nd-best in the row)
                # argpartition is O(N), safer for larger N_active
                part = np.argpartition(row, -2)[-2:]
                # ensure loser != winner
                if w_local in part:
                    l_local = int(part[0] if part[1] == w_local else part[1])
                else:
                    # fallback if argpartition didn't include winner (edge case)
                    l_local = int(part[0])

                gap = float(row[w_local] - row[l_local])
                if gap < self.margin_tau:
                    continue

                # map back to GLOBAL indices for the policy head
                winners.append(active_idx[w_local])
                losers.append(active_idx[l_local])
                margins.append(gap)

            if not winners:
                continue  # nothing to update on this iteration

            # One DPO update; returns preference loss
            loss = float(self.dpo.update(winners, losers))

            # Policy diagnostics (AFTER the update)
            with torch.no_grad():
                logits = self.dpo.policy.logits            # (N_total,)
                logp   = logits.log_softmax(-1)
                p      = logp.exp()
                entropy = float(-(p * logp).sum().item())

                kl_ref = None
                if (not self.dpo.reference_free) and (self.dpo._ref_logits is not None):
                    logp_ref = self.dpo._ref_logits.log_softmax(-1)
                    kl_ref = float((p * (logp - logp_ref)).sum().item())

            # Policy-expected BEM prob on the ACTIVE set for this batch
            p_active = p[active_idx].cpu().numpy()  # (N_active,)
            exp_prob = float((scores_mat @ p_active).mean())

            # Log history
            self.history["loss"].append(loss)
            self.history["pairs"].append(len(winners))
            self.history["margin"].append(float(np.mean(margins)) if margins else None)
            self.history["entropy"].append(entropy)
            self.history["exp_prob"].append(exp_prob)
            self.history["kl_ref"].append(kl_ref)

        # Return πθ(a) for ranking outside
        return self.dpo.get_action_preferences().tolist()



def get_evaluator(name):
    name = name.lower()
    if name in {"ucb", "ucb-e"}:
        return UCBBanditEvaluator
    if name in {"sr", "s-sr"}:
        return SuccessiveRejectsEvaluator
    if name == "sh":
        return SuccessiveHalvingEvaluator
    if name == "bf":
        return BruteForceEvaluator
    if name == "ppo":
        return PPOEvaluator
    if name == "dpo":
        return DPOEvaluator
    raise ValueError(f"Unknown evaluator: {name}")