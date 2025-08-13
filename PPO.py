import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.__init__() 

class ActorCritic(nn.Module):
    # MLP actor–critic: one network outputs the action probabilities, another outputs the scalar state-value.

    def __init__(self, state_dim, action_dim, hidden: int = 64):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1))
        
    def act(self, state, memory):
        if state is None:
            state = np.zeros(1, dtype=np.float32)
        state = torch.tensor(state, dtype=torch.float32, device=device)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()
    

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state).squeeze(-1)

        return action_logprobs, state_value, dist_entropy
    

class PPO:
    def __init__(self, state_dim: int = 1, 
                       action_dim: int = 64, 
                       hidden: int = 64,
                       lr: float = 2e-3, 
                       betas: tuple = (0.9, 0.999),
                       gamma: float = 0.99, 
                       K_epochs: int = 4, 
                       eps_clip: float = 0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory: Memory):
        return self.policy_old.act(state, memory)
    
    def update(self, memory):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_rewards = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_rewards = 0
            discounted_rewards = reward + (self.gamma * discounted_rewards)
            rewards.insert(0, discounted_rewards)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # Safeguard to avoid dividing by zero when std becomes zero due to identical rewards
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        else:
            rewards = rewards - rewards.mean()
        
        # trajectories to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # optimization loop for K epochs, uses cliped gradient ascent
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


    @torch.no_grad()
    def get_action_preferences(self):
        # Return probabilities for the dummy zero-state.
        # Useful for ranking prompts without extra sampling noise.
        
        zero_state = torch.zeros(1, dtype=torch.float32, device=device)
        probs      = self.policy.actor(zero_state)
        return probs.squeeze(0).cpu().numpy()

    def save(self, path: str):
        # Save trainable policy parameters to path
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        # Load parameters from path into both actor/critic copies
        sd = torch.load(path, map_location=device)
        self.policy.load_state_dict(sd)

        self.policy_old.load_state_dict(sd)
