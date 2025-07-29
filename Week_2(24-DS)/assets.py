import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
 
# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition을 저장하기 위한 namedtuple입니다! 아래 ReplayMemory 클래스에서 사용해요. 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

'''
gymnasium은 stable_baseline으로 간단하게 구현할 수 있지만, 
상황에 따라 직접 DQN과 같은 Agent의 구성요소를 구현할 수도 있고,
Custom Environment를 만들어서 사용할 수도 있습니다. 

DQN에선 Action 후 State, action, next_state, reward 등을 저장하는 ReaplyMemory를 사용하고, 
Q-network를 신경망으로 근사해 사용합니다. 또한 Target Network와 Q - Network를 구분해 학습을 안정화시킵니다. 

이번 과제에선 env를 구현하진 않고, DQN의 구성요소들 중 Q-network와 ReplayMemory를 구현해보겠습니다.

ReplayMemory는 deque를 사용해 구현하면 되고, 

DQN은 nn.Module을 상속받아 구현하시면 됩니다. 필요한 메소드는 정해두었으니 참고하시면 됩니다! 
'''

####### 여기서부터 코드를 작성하세요 #######
# ReplayMemory 클래스를 구현해주세요!
class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Buffer 초기화
        Args:
            capacity: 버퍼의 최대 크기
        """
        # TODO: deque를 사용해서 최대 크기가 capacity인 메모리 버퍼를 만드세요
        # Hint: self.memory = deque([], maxlen=capacity)
        pass

    def push(self, *args):
        """Transition 저장"""
        # TODO: Transition(*args)를 생성해서 메모리에 추가하세요
        # Hint: self.memory.append(Transition(*args))
        pass

    def sample(self, batch_size):
        """
        랜덤하게 배치 크기만큼 샘플링
        Args:
            batch_size: 샘플링할 배치 크기
        Returns:
            무작위로 선택된 transition들의 리스트
        """
        # TODO: random.sample을 사용해서 메모리에서 batch_size만큼 랜덤 샘플링하세요
        # Hint: return random.sample(self.memory, batch_size)
        pass

    def __len__(self):
        """현재 메모리에 저장된 transition 개수 반환"""
        # TODO: 메모리 길이를 반환하세요
        # Hint: return len(self.memory)
        pass
    

# DQN 모델을 구현해주세요! Atari Game에선 CNN 모듈을 사용하지만, 구현은 간단하게 MLP로 해도 됩니다. 성능을 비교해보며 자유로이 구현해보세요! 
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, is_atari=False):
        """
        DQN 네트워크 초기화
        Args:
            n_observations: 관측 공간의 크기 (Lunar Lander: 8, Atari: (4, 84, 84))
            n_actions: 행동 공간의 크기
            is_atari: Atari 환경인지 여부 (CNN vs MLP 결정)
        """
        super(DQN, self).__init__()
        self.is_atari = is_atari
        
        if is_atari:
            # TODO: Atari용 CNN 구조를 구현하세요 (Nature DQN 논문 기반)
            # Hint: Conv2d → ReLU → Conv2d → ReLU → Conv2d → ReLU
            # 구체적인 구조:
            # Conv2d(4, 32, kernel_size=8, stride=4) → ReLU
            # Conv2d(32, 64, kernel_size=4, stride=2) → ReLU  
            # Conv2d(64, 64, kernel_size=3, stride=1) → ReLU
            
            self.features = nn.Sequential(
                # TODO: 3개의 Conv2d 레이어와 ReLU 활성화 함수를 추가하세요
            )
            
            # TODO: CNN 출력을 Fully Connected로 연결하는 head 부분을 구현하세요
            # Hint: Flatten() → Linear(3136, 512) → ReLU → Linear(512, n_actions)
            # 참고: 마지막 Conv2d 출력 크기는 64 * 7 * 7 = 3136입니다
            self.head = nn.Sequential(
                # TODO: Flatten, Linear, ReLU, Linear 레이어를 추가하세요
            )
        else:
            # TODO: Lunar Lander용 MLP 구조를 구현하세요
            # Hint: Linear(n_observations, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, n_actions)
            self.network = nn.Sequential(
                # TODO: 3개의 Linear 레이어와 2개의 ReLU 활성화 함수를 추가하세요
            )

    def forward(self, x):
        """
        순전파
        Args:
            x: 입력 상태 (Lunar Lander: [batch, 8], Atari: [batch, 4, 84, 84])
        Returns:
            각 행동에 대한 Q값들 [batch, n_actions]
        """
        if self.is_atari:
            # TODO: Atari 순전파를 구현하세요
            # Hint: 
            # 1. 픽셀 값을 0-1로 정규화: x = x.float() / 255.0
            # 2. CNN features 통과: x = self.features(x)
            # 3. head 통과해서 Q값 반환: return self.head(x)
            pass
        else:
            # TODO: Lunar Lander 순전파를 구현하세요
            # Hint: return self.network(x)
            pass

####### 여기까지 코드를 작성하세요 #######


class DQNAgent:
    def __init__(self, state_size, action_size, eps_start, eps_end, eps_decay, gamma, lr, batch_size, tau, is_atari=False):
        self.state_size = state_size
        self.action_size = action_size
        self.is_atari = is_atari
        
        # Atari의 경우 더 큰 replay memory 사용
        memory_size = 100000 if is_atari else 10000
        self.memory = ReplayMemory(memory_size)
        
        self.policy_net = DQN(state_size, action_size, is_atari).to(device)
        self.target_net = DQN(state_size, action_size, is_atari).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.episode_rewards = []
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_net()

        # 플로팅 초기화
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label='Total Reward')
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_rewards(self):
        self.reward_line.set_xdata(range(len(self.episode_rewards)))
        self.reward_line.set_ydata(self.episode_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
def preprocess_atari_state(observation):
    """
    Atari 관측값을 DQN에 입력할 수 있는 형태로 전처리
    """
    import numpy as np
    import torch

    if not isinstance(observation, np.ndarray):
        observation = np.asarray(observation)  # LazyFrames → ndarray

    observation = torch.tensor(observation, dtype=torch.float32, device=device) / 255.0
    observation = observation.unsqueeze(0)  # 배치 차원 추가 → [1, 4, 84, 84]
    return observation