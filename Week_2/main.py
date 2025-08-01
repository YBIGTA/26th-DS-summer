import argparse
import gymnasium as gym
import torch
from itertools import count
from tqdm import tqdm
from assets import DQNAgent, device, preprocess_atari_state
import shimmy

# gym.register_envs(ale_py)

def create_environment(env_id, render_mode='rgb_array'):
    """
    환경 생성 및 Atari 환경인 경우 전처리 래퍼 적용
    """
    if 'ALE/' in env_id or 'Atari' in env_id:
        # Atari 환경인 경우
        try:
            from gymnasium.wrappers import AtariPreprocessing
            from gymnasium.wrappers import FrameStack

            # from gymnasium.wrappers import FrameStack  
            base_env = gym.make(env_id, render_mode=render_mode, frameskip=1)

            # base_env = gym.make(env_id, render_mode=render_mode)
            # Atari 전처리: 그레이스케일, 리사이즈, 프레임 스킵 등
            env = FrameStack(
                AtariPreprocessing(
                    base_env, 
                    scale_obs=True,  # 84x84로 리사이즈
                    grayscale_obs=True,  # 그레이스케일 변환
                    terminal_on_life_loss=True  # 생명 잃으면 에피소드 종료
                ), 
                4  # 4프레임 스택
            )
            return env, True  # is_atari=True
        except ImportError:
            print("Atari 환경을 위해 'pip install gymnasium[atari,accept-rom-license]'를 설치하세요.")
            raise
    else:
        # Lunar Lander 등 일반 환경
        env = gym.make(env_id, render_mode=render_mode)
        return env, False  # is_atari=False

def main(args):
    # 환경 생성
    env, is_atari = create_environment(args.env_id, args.render_mode)
    
    if is_atari:
        # Atari 환경: 상태는 (4, 84, 84) 프레임 스택
        state_size = env.observation_space.shape  # (4, 84, 84)
    else:
        # Lunar Lander: 상태는 8차원 벡터
        state_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    # Atari용 하이퍼파라미터 조정
    if is_atari:
        eps_decay = args.eps_decay * 10  # 더 천천히 탐험 감소
        batch_size = 32  # Atari 논문에서 사용하는 배치 크기
        target_update = 1000  # 더 자주 타겟 네트워크 업데이트
    else:
        eps_decay = args.eps_decay
        batch_size = args.batch_size
        target_update = args.target_update

    agent = DQNAgent(
        state_size, action_size, args.eps_start, args.eps_end, 
        eps_decay, args.gamma, args.lr, batch_size, args.tau, is_atari
    )

    print(f"환경: {args.env_id}")
    print(f"Atari 환경: {is_atari}")
    print(f"상태 크기: {state_size}")
    print(f"행동 크기: {action_size}")

    for i_episode in tqdm(range(args.episodes)):
        observation, info = env.reset()
        
        if is_atari:
            state = preprocess_atari_state(observation)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done and terminated:
                next_state = None
            else:
                if is_atari:
                    next_state = preprocess_atari_state(observation)
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)
            state = next_state

            agent.optimize_model()

            # Soft update of target network
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                if is_atari:
                    print(f"Episode {i_episode}: Total reward: {total_reward}")
                else:
                    if terminated and total_reward >= 200:  # 성공적인 착지로 간주
                        print(f"Episode {i_episode}: Successful landing! Total reward: {total_reward}")
                    else:
                        print(f"Episode {i_episode}: Crash or failure. Total reward: {total_reward}")
                
                agent.episode_rewards.append(total_reward)
                agent.plot_rewards()
                break

        # 주기적으로 타겟 네트워크 완전 업데이트
        if i_episode % target_update == 0:
            agent.update_target_net()

    print('Complete')

    # 모델 저장
    agent.policy_net.to('cpu')
    print('now save!')
    torch.save(agent.policy_net.state_dict(), args.save_path)
    agent.policy_net.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='ALE/Breakout-v5', 
                        help='Environment ID (e.g., LunarLander-v2, ALE/Breakout-v5, ALE/Pong-v5)')
    parser.add_argument('--render_mode', type=str, default='rgb_array', 
                        help='Render mode (human, rgb_array, None)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train the agent')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Starting value of epsilon for epsilon-greedy policy')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Ending value of epsilon for epsilon-greedy policy')
    parser.add_argument('--eps_decay', type=int, default=200, help='Epsilon decay factor')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--target_update', type=int, default=10, help='Number of episodes between target network updates')
    parser.add_argument('--save_path', type=str, default='dqn_lunarlander.pth', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)
