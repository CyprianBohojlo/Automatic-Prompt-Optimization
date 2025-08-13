from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List



# Policy network

class Policy(nn.Module):
    def __init__(self, n_actions: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        # cached dummy zero-state
        self.register_buffer("_state", torch.zeros(1, n_actions))

    # raw logits (shape = n_actions)
    def forward(self) -> torch.Tensor:
        return self.actor(self._state).squeeze(0)

    # helpers
    def log_probs(self) -> torch.Tensor:
        return self.forward().log_softmax(-1)

    def probs(self) -> torch.Tensor:
        return self.forward().softmax(-1)



#  Preference loss 

def preference_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
    reference_free: bool = True,
) -> torch.Tensor:
    if reference_free:
        ref_chosen = ref_rejected = torch.zeros_like(chosen_logp)

    logits = beta * ((chosen_logp - ref_chosen) -
                     (rejected_logp - ref_rejected))
    return F.binary_cross_entropy_with_logits(logits,
                                              torch.ones_like(logits))



#  trainer wrapper 

class DPO:
    
   # Discrete-action DPO trainer.
    # update(winners, losers)  – single SGD step
    # get_action_preferences() – returns πθ(a) as 1-D tensor
    

    def __init__(self,
                 n_actions: int,
                 beta: float = 0.1,
                 lr: float = 3e-4,
                 hidden: int = 128,
                 reference_free: bool = True):
        self.beta = beta
        self.reference_free = reference_free
        self.policy = Policy(n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._ref_logits = None            # frozen π_ref

    # single optimisation step
    def update(self, winners: List[int], losers: List[int]) -> None:
        w = torch.tensor(winners)
        l = torch.tensor(losers)

        logp = self.policy.log_probs()             

        if self.reference_free:
            ref_logp = torch.zeros_like(logp)
        else:
            if self._ref_logits is None:
                self._ref_logits = self.policy.forward().detach()
            ref_logp = self._ref_logits.log_softmax(-1)

        loss = preference_loss(
            chosen_logp   = logp[w],
            rejected_logp = logp[l],
            ref_chosen    = ref_logp[w],
            ref_rejected  = ref_logp[l],
            beta          = self.beta,
            reference_free= self.reference_free)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # softmax over actions – used for beam pruning
    def get_action_preferences(self) -> torch.Tensor:
        return self.policy.probs()

