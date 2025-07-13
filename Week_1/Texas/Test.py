import numpy as np
import pickle
import random
import time
from collections import defaultdict
from pettingzoo.classic import texas_holdem_v4

# --- Q-테이블 파일 경로 (실제 경로로 수정) ---
# PLAYER1_Q_PATH = 'texas_holdem_sarsa_3_player_1.pkl'
# PLAYER0_Q_PATH = 'texas_holdem_sarsa_2_player_0.pkl'

PLAYER0_Q_PATH = 'C:/Users/PC/GitHub/26th-DS-summer/Week_1/Texas/texas_holdem_sarsa_base_player_0.pkl'
PLAYER1_Q_PATH = 'C:/Users/PC/GitHub/26th-DS-summer/Week_1/Texas/texas_holdem_q_learning_base_player_1.pkl'


# --- 환경 참조해서 action_space 크기 가져오기 ---
env = texas_holdem_v4.env(render_mode=None)
env.reset()
ACTION_N = {
    'player_0': env.action_space('player_0').n,
    'player_1': env.action_space('player_1').n
}
env.close()

# --- Q-테이블 로드 & defaultdict 래핑 ---
with open(PLAYER0_Q_PATH, 'rb') as f:
    raw_q0 = pickle.load(f)
with open(PLAYER1_Q_PATH, 'rb') as f:
    raw_q1 = pickle.load(f)

Q_TABLE = {
    'player_0': defaultdict(lambda: np.zeros(ACTION_N['player_0']), raw_q0),
    'player_1': defaultdict(lambda: np.zeros(ACTION_N['player_1']), raw_q1)
}

# --- 상태 키 생성 함수 ---
def obs_to_key(obs, agent):
    # agent 구분 위해 튜플 앞에 agent 이름 포함
    return (agent,) + tuple(obs['observation'].flatten())

# --- Greedy 정책으로 행동 선택 ---
def choose_action(agent, obs):
    table = Q_TABLE[agent]
    key   = obs_to_key(obs, agent)
    legal = [i for i, ok in enumerate(obs['action_mask']) if ok]
    qv    = table[key]
    # legal 액션 중 최대 Q값 찾기
    max_q = max(qv[a] for a in legal)
    # tie-breaker: 같은 최대값 액션들 중 랜덤 선택
    best = [a for a in legal if qv[a] == max_q]
    return random.choice(best)

# --- 한 핸드 플레이 (render=True 면 매 스텝 렌더링) ---
def play_hand(render=False):
    env = texas_holdem_v4.env(render_mode='human' if render else None)
    obs = env.reset()
    flag= False  # 핸드 종료 플래그
    if render:
        env.render()
    rewards = {a: 0 for a in env.agents}

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        #print(agent,reward)
        rewards[agent] += reward

        action = None if (term or trunc) else choose_action(agent, obs)
        env.step(action)

        if render:
            env.render()
            time.sleep(0.2)

        if term or trunc:
            if flag:
                break
            else:
                flag = True
        
    if render:
        input("▶ 이 페이즈 종료. 엔터를 누르면 다음 핸드를 진행합니다.")
            
    env.close()
    return rewards['player_0'], rewards['player_1']

# --- 한 매치 (스택이 0 될 때까지 여러 핸드 반복) 플레이 ---
def play_match(initial_stack=10, render=False):
    stacks = {'player_0': initial_stack, 'player_1': initial_stack}

    while stacks['player_0'] > 0 and stacks['player_1'] > 0:
        # render=True 면 모든 핸드를 GUI로 확인
        r0, r1 = play_hand(render=render)
        stacks['player_0'] += r0
        stacks['player_1'] += r1
        if render:
            print(f"  ▶ 핸드 결과 P0:{r0:+}, P1:{r1:+} | 스택 P0:{stacks['player_0']}, P1:{stacks['player_1']}")

    # 동시에 0 이면 무승부
    if stacks['player_0'] <= 0 and stacks['player_1'] <= 0:
        return 'draw'
    return 'player_0' if stacks['player_1'] <= 0 else 'player_1'

# --- 메인: 여러 매치 반복, 마지막 매치는 시각화 ---
if __name__ == '__main__':
    NUM_MATCHES   = 100
    INITIAL_STACK = 10
    results = {'player_0': 0, 'player_1': 0, 'draw': 0}

    for i in range(1, NUM_MATCHES + 1):
        is_last = (i == NUM_MATCHES)
        print(f"\n=== Match {i} 시작 {'(시각화)' if is_last else '(시뮬레이션)'} ===")
        winner = play_match(INITIAL_STACK, render=is_last)
        results[winner] += 1
        print(f"=== Match {i} 결과: {winner} 승리 ===")

    print("\n--- 전체 매치 결과 ---")
    print(f"player_0 wins: {results['player_0']}")
    print(f"player_1 wins: {results['player_1']}")
    print(f"draws         : {results['draw']}")
