import numpy as np
import pickle
import random
from collections import defaultdict
from pettingzoo.classic import texas_holdem_v4
from tqdm import tqdm

# 1) 환경 생성 (render_mode="human" 으로 시각화 가능)
env = texas_holdem_v4.env(render_mode=None)
env.reset()
agents = env.agents  # ['player_0', 'player_1']

# 2) Q-table 초기화: 각 에이전트별로 SARSA Q-table 하나씩
q_tables = {
    agent: defaultdict(lambda: np.zeros(env.action_space(agent).n))
    for agent in agents
}

# 3) 하이퍼파라미터
alpha       = 0.01    # 학습률
gamma       = 0.9    # 할인율
epsilon     = 0.2     # 탐험률
num_episodes = 100 # 학습 에피소드 수

# 4) 관측(observation)을 Q-table 키로 변환하는 함수
def obs_to_key(obs, agent):
    """
    - agent를 포함시켜 player_0/player_1을 구분
    - observation 벡터를 flatten 후 튜플로 변환
    """
    return (agent,) + tuple(obs["observation"].flatten())

# 5) ε-greedy 행위 선택 함수
def choose_action(q_table, state_key, legal_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(legal_actions)
    qv = q_table[state_key]
    # legal_actions 에 해당하는 값만 남기고 나머진 -inf
    masked = np.full_like(qv, -np.inf)
    for a in legal_actions:
        masked[a] = qv[a]
    return int(np.argmax(masked))

# 6) 학습 루프
for episode in tqdm(range(num_episodes), desc="SARSA Training"):
    env.reset()
    # 에피소드당 이전(상태, 행동, 보상) 저장
    prev = {agent: {"state": None, "action": None, "reward": 0} for agent in agents}
    flag= False
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        legal = [i for i, ok in enumerate(obs["action_mask"]) if ok]

        # 종료된 상태면 None 액션
        if term or trunc:
            action = None
        else:
            # (1) 상태 키 생성
            state_key = obs_to_key(obs, agent)

            # (2) 이전 스텝이 있으면 SARSA 업데이트
            if prev[agent]["state"] is not None:
                next_action = choose_action(
                    q_tables[agent],
                    state_key,
                    legal,
                    epsilon
                )
                td_target = None ### 여기를 바꿔주세요 ###
                td_error  = None ### 여기를 바꿔주세요 ###
                q_tables[agent][prev[agent]["state"]][prev[agent]["action"]] += alpha * td_error

                action = next_action
            else:
                # 첫 번째 스텝인 경우
                action = choose_action(q_tables[agent], state_key, legal, epsilon)

            # (3) prev 업데이트
            prev[agent] = {"state": state_key, "action": action, "reward": reward}

        # (4) 환경에 액션 적용
        env.step(action)

        # (5) 터미널 보상 업데이트 후 반복 종료
        if term or trunc:
            if prev[agent]["state"] is not None:
                td_target = None ### 여기를 바꿔주세요 ###
                td_error  = None ### 여기를 바꿔주세요 ###
                q_tables[agent][prev[agent]["state"]][prev[agent]["action"]] += alpha * td_error
            if flag:
                break
            else:
                flag=True
        


# 7) 학습된 Q-table 저장
for agent in agents:
    # --- 학습 결과 통계 ---
    all_q = np.vstack(list(q_tables[agent].values()))
    print("총 상태 수:", len(q_tables[agent]))
    print("Q값 최소:", np.min(all_q))
    print("Q값 최대:", np.max(all_q))
    print("Q값 평균:", np.mean(all_q))
    fname = f"texas_holdem_sarsa_4_{agent}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(dict(q_tables[agent]), f)
    print(f"Saved {agent} Q-table to {fname}")
