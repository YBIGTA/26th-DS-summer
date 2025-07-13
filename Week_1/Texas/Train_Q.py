import numpy as np
import pickle
import random
from collections import defaultdict
from pettingzoo.classic import texas_holdem_v4
from tqdm import tqdm

# 1) 환경 생성 (render_mode=None: 학습용)
env = texas_holdem_v4.env(render_mode=None)
env.reset()
agents = env.agents  # ['player_0', 'player_1']

# 2) Q-table 초기화: 에이전트별 Q-table
q_tables = {
    agent: defaultdict(lambda: np.zeros(env.action_space(agent).n))
    for agent in agents
}

env.close()

# 3) 하이퍼파라미터
alpha        = 0.01    # 학습률
gamma        = 0.95    # 할인율
epsilon      = 0.3     # 탐험률
num_episodes = 100   # 학습 에피소드 수

# 4) 상태(observation) -> Q-table 키 변환 함수
def obs_to_key(obs, agent):
    return (agent,) + tuple(obs["observation"].flatten())

# 5) ε-greedy 행위 선택 함수
def choose_action(q_table, state_key, legal_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(legal_actions)
    qv = q_table[state_key]
    masked = np.full_like(qv, -np.inf)
    for a in legal_actions:
        masked[a] = qv[a]
    return int(np.argmax(masked))

# 6) Q-learning 학습 루프
for ep in tqdm(range(1, num_episodes+1), desc="Q-learning Training"):
    env = texas_holdem_v4.env()
    env.reset()
    flag=False

    # 에피소드마다 이전 상태/행동/보상 저장
    prev = {agent: {"state": None, "action": None, "reward": 0} for agent in agents}

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        legal = [i for i, ok in enumerate(obs["action_mask"]) if ok]
        if term or trunc:
            action = None
        else:
            state_key = obs_to_key(obs, agent)

            # 이전 스텝에 대해 Q-learning 업데이트
            if prev[agent]["state"] is not None:
                max_next_q = np.max([q_tables[agent][state_key][a] for a in legal]) if legal else 0
                td_target  = None ### 여기를 바꿔주세요 ###ㄴ
                td_error   = None ### 여기를 바꿔주세요 ###
                q_tables[agent][prev[agent]["state"]][prev[agent]["action"]] += alpha * td_error

            # 현재 상태에서 행동 선택
            action = choose_action(q_tables[agent], state_key, legal, epsilon)
            # 다음 업데이트를 위해 저장
            prev[agent] = {"state": state_key, "action": action, "reward": reward}

        env.step(action)

        # 터미널 보상 업데이트 및 에피소드 종료
        if term or trunc:
            if prev[agent]["state"] is not None:
                td_target = None ### 여기를 바꿔주세요 ###
                td_error  = None ### 여기를 바꿔주세요 ###
                q_tables[agent][prev[agent]["state"]][prev[agent]["action"]] += alpha * td_error
            if flag:
                break
            else:
                flag=True
        

    env.close()

# 7) 학습 결과 저장
for agent in agents:
    # --- 학습 결과 통계 ---
    all_q = np.vstack(list(q_tables[agent].values()))
    print("총 상태 수:", len(q_tables[agent]))
    print("Q값 최소:", np.min(all_q))
    print("Q값 최대:", np.max(all_q))
    print("Q값 평균:", np.mean(all_q))
    fname = f"texas_holdem_q_learning_{agent}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(dict(q_tables[agent]), f)
    print(f"Saved Q-table for {agent} to {fname}")
