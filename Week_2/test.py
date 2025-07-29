import argparse
import gymnasium as gym
import torch
from assets_hj import DQN, device, preprocess_atari_state
import warnings
import shimmy

# Deprecation warning 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_environment(env_id, render_mode='human'):
    """
    환경 생성 및 Atari 환경인 경우 전처리 래퍼 적용
    """
    if 'ALE/' in env_id or 'Atari' in env_id:
        # Atari 환경인 경우
        try:
            from gymnasium.wrappers import AtariPreprocessing
            from gymnasium.wrappers import FrameStack
                    
            base_env = gym.make(env_id, render_mode=render_mode, frameskip=1)
            env = FrameStack(
                AtariPreprocessing(
                    base_env, 
                    scale_obs=True,
                    grayscale_obs=True,
                    terminal_on_life_loss=True
                ), 
                4
            )
            return env, True
        except ImportError as e:
            print(f"Atari 환경 설정 중 오류: {e}")
            print("다음 명령어로 Atari 패키지를 설치하세요:")
            print("pip install 'gymnasium[atari,accept-rom-license]'")
            print("또는:")
            print("pip install ale-py autorom[accept-rom-license]")
            raise
    else:
        env = gym.make(env_id, render_mode=render_mode)
        return env, False

def test(args):
    # 환경 생성
    env, is_atari = create_environment(args.env_id, args.render_mode)
    
    if is_atari:
        state_size = env.observation_space.shape
    else:
        state_size = env.observation_space.shape[0]
    
    action_size = env.action_space.n
    
    # 모델 로드
    policy_net = DQN(state_size, action_size, is_atari).to(device)
    policy_net.load_state_dict(torch.load(args.model_path, map_location=device))
    policy_net.eval()

    total_rewards = []

    print(f"테스트 환경: {args.env_id}")
    print(f"Atari 환경: {is_atari}")
    print(f"상태 크기: {state_size}")
    print(f"행동 크기: {action_size}")

    for i_episode in range(args.test_episodes):
        observation, info = env.reset()
        
        if is_atari:
            state = preprocess_atari_state(observation)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward

            if terminated or truncated:
                done = True
            else:
                if is_atari:
                    state = preprocess_atari_state(observation)
                else:
                    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(total_reward)
        print(f"Episode {i_episode}: Total reward: {total_reward}")

    avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
    print(f'Average reward over last {min(10, len(total_rewards))} episodes: {avg_reward}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='LunarLander-v2',
                        help='Environment ID (e.g., LunarLander-v2, ALE/Breakout-v5)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes to test the agent')
    parser.add_argument('--render_mode', type=str, default='human', help='Render mode')
    args = parser.parse_args()
    test(args)