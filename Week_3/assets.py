import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, List

# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn=F.tanh):
        super().__init__()
        '''
        TODO: 정책 신경망을 구현하세요! 원래는 두개 다 비우려고 했는데... 생각해보니까 아직 Actor Critic을 하지 않아서... 신경망을 만드는 건 다들 잘 하시니까~
        h1, h2 = 
        self.input_layer = 
        self.hidden_layers = 
        self.mu_layer = 
        self.log_std_layer = 
        self.activation_fn = 
        '''

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn=F.tanh):
        super().__init__()
        '''
        TODO: 가치 신경망을 구현하세요! 
        h1, h2 = 
        self.input_layer = 
        self.hidden_layers = 
        self.output_layer = 
        self.activation_fn = 
        '''
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        return x

# Simple Experience Buffer (FIFO of transitions)
class ExperienceBuffer:
    def __init__(self):
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []

    def store(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        # TODO: One line
        pass


    def sample(self) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        '''
        TODO:
        저장된 전이들을 배치 형태의 Torch 텐서로 변환합니다: 상태(states), 행동(actions), 보상(rewards), 다음 상태(next_states), 종료 여부(dones)
        '''
        # clear after sampling (on-policy)
        self.buffer.clear()
        return states, actions, rewards, next_states, dones

    @property
    def size(self) -> int:
        return len(self.buffer)

# A2C Algorithm (synchronous, batched)
class A2C:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (64, 64),
        activation_fn=F.relu,
        n_steps: int = 2048,
        batch_size: int = 64,
        policy_lr: float = 3e-4,
        value_lr: float = 3e-4,
        gamma: float = 0.99,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dims, activation_fn).to(self.device)

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=value_lr)

        self.buffer = ExperienceBuffer()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor]:
        self.policy.train(training)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mu, std = self.policy(state_t)
        dist = Normal(mu, std)
        z = dist.sample() if training else mu
        action = torch.tanh(z)
        return action.cpu().numpy(), dist.log_prob(z).sum(dim=-1, keepdim=True)

    def _compute_returns_and_advantages(
        self, states: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            values = self.value(states.to(self.device))
            next_values = self.value(next_states.to(self.device))
            if self.use_gae:
                deltas = rewards.to(self.device) + (1 - dones.to(self.device)) * self.gamma * next_values - values
                adv = torch.zeros_like(deltas, device=self.device)
                running = torch.zeros(1, 1, device=self.device)
                for t in reversed(range(len(rewards))):
                    running = deltas[t] + (1 - dones[t]) * self.gamma * self.gae_lambda * running
                    adv[t] = running
                returns = values + adv
            else:
                returns = torch.zeros_like(rewards, device=self.device)
                running = torch.zeros(1, 1, device=self.device)
                for t in reversed(range(len(rewards))):
                    running = rewards[t].to(self.device) + (1 - dones[t]) * self.gamma * (running if t < len(rewards) - 1 else next_values[t])
                    returns[t] = running
                adv = returns - values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return returns.detach(), adv.detach()

    def update(self) -> None:
        self.policy.train()
        self.value.train()

        states, actions, rewards, next_states, dones = self.buffer.sample()
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), [states, actions, rewards, next_states, dones])

        returns, advantages = self._compute_returns_and_advantages(states, rewards, next_states, dones)

        dataset = TensorDataset(states, actions, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for s_batch, a_batch, ret_batch, adv_batch in loader:
            value_pred = self.value(s_batch)
            value_loss = F.mse_loss(value_pred, ret_batch)

            mu, std = self.policy(s_batch)
            dist = Normal(mu, std)

            atanh_a = torch.atanh(torch.clamp(a_batch, -0.999999, 0.999999))
            '''
            TODO: 이부분도 해주세요. 마지막 TODO입니다!

            log_prob = 주어진 행동(atanh_a)에 대해 정책 dist로부터 log π(a|s)를 계산하고, 액션 차원에 대해 합산합니다.

            entropy = 정책의 엔트로피를 계산하고(액션 차원 합산 후, 배치 평균), 탐험(exploration)을 촉진하기 위해 사용합니다.
            
            policy_loss = −(log_prob*adv_batch).mean() 형태로 계산합니다. 
            loss = Actor,Critic,Entropy
            '''

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()

    def step(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        self.buffer.store(transition)
        if self.buffer.size >= self.n_steps:
            self.update()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
