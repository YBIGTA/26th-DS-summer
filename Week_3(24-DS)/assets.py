import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, List

# 정책 신경망 (Policy Network)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn: nn.Module = F.tanh):
        super(PolicyNetwork, self).__init__()
        ########################################
        '''TODO: 정책 신경망을 구현하세요! 원래는 두개 다 비우려고 했는데... 생각해보니까 아직 Actor Critic을 하지 않아서... 신경망을 만드는 건 다들 잘 하시니까~
        self.input_layer = 
        self.hidden_layers = 
        self.mu_layer = 
        self.log_std_layer = 
        self.activation_fn = activation_fn
        '''
        ########################################
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

# 가치 신경망 (Value Network)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn: nn.Module = F.tanh):
        super(ValueNetwork, self).__init__()
        ########################################
        '''TODO: 가치 신경망을 구현하세요! 
        self.input_layer = 
        self.hidden_layers = 
        self.output_layer = 
        self.activation_fn = 
        '''
        ########################################
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        return x

# 경험 버퍼 (Experience Buffer)
# 지난번에 보았던 Buffer와 같은 기능을 합니다! 

class ExperienceBuffer:
    def __init__(self):
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []

    def store(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        '''TODO'''
        pass
    def sample(self) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        '''TODO'''

        pass

    @property
    def size(self) -> int:
        return len(self.buffer)

# PPO 알고리즘 (PPO Algorithm)
# PPO의 장점은 TRPO와 다르게 간단하게 clipping으로 구현이 가능하다는 점에 있습니다. 아래를 잘보고 빈칸을 잘 채워주세요! 
class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (64, 64),
        activation_fn: nn.Module = F.relu,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        policy_lr: float = 0.0003,
        value_lr: float = 0.0003,
        gamma: float = 0.99,
        lmda: float = 0.95,
        clip_ratio: float = 0.2,
        vf_coef: float = 1.0,
        ent_coef: float = 0.01,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dims, activation_fn).to(self.device)
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lmda = lmda
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=value_lr)
        
        self.buffer = ExperienceBuffer()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor]:
        self.policy.train(training)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mu, std = self.policy(state)
        dist = Normal(mu, std)
        z = dist.sample() if training else mu
        action = torch.tanh(z)
        return action.cpu().numpy(), dist.log_prob(z).sum(dim=-1, keepdim=True)

    def update(self) -> None:
        self.policy.train()
        self.value.train()
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), [states, actions, rewards, next_states, dones])
        
        with torch.no_grad():
            deltas = rewards + (1 - dones) * self.gamma * self.value(next_states) - self.value(states)
            advantages = torch.zeros_like(deltas).to(self.device)
            returns = torch.zeros_like(rewards).to(self.device)
            acc_advantage = 0
            acc_return = 0
            for t in reversed(range(len(rewards))):
                acc_advantage = deltas[t] + self.gamma * self.lmda * acc_advantage * (1 - dones[t])
                acc_return = rewards[t] + self.gamma * acc_return * (1 - dones[t])
                advantages[t] = acc_advantage
                returns[t] = acc_return
            
            mu, std = self.policy(states)
            dist = Normal(mu, std)
            log_prob_old = dist.log_prob(torch.atanh(actions)).sum(dim=-1, keepdim=True)
        
        dataset = TensorDataset(states, actions, returns, advantages, log_prob_old)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.n_epochs):
            for batch in loader:
                s_batch, a_batch, r_batch, adv_batch, log_prob_old_batch = batch
                    
                value_loss = F.mse_loss(self.value(s_batch), r_batch)
                mu, std = self.policy(s_batch)
                dist = Normal(mu, std)
                log_prob = dist.log_prob(torch.atanh(a_batch)).sum(dim=-1, keepdim=True)
                ratio = (log_prob - log_prob_old_batch).exp()

                '''TODO : 이부분을 채워주세요! Hint : Clipping을 사용해주세요~
                surr1 = 
                surr2 = 
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = dist.entropy().mean()
                loss = 
                '''
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

    def step(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        self.buffer.store(transition)
        if self.buffer.size >= self.n_steps:
            self.update()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
