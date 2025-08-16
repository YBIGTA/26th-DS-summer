import argparse
import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from assets import A2C  

ENV_IDS = {
    "mountaincar": "MountainCarContinuous-v0",
    "hopper": "Hopper-v4",
}

def get_activation_fn(name: str):
    name = (name or "tanh").lower()
    if name == "tanh":
        return F.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation_fn: {name}")

def load_config(path: str):
    # If no config.json file located, the code will run with these configurations. 
    cfg = {
        "hidden_dims": [64, 64],
        "activation_fn": "tanh",
        "n_steps": 2048,
        "batch_size": 64,
        "policy_lr": 0.0003,
        "value_lr": 0.0003,
        "gamma": 0.99,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
    }
    try:
        with open(path, "r") as f:
            on_disk = json.load(f)
        cfg.update(on_disk)
    except FileNotFoundError:
        pass
    return cfg

def make_env(env_key: str, seed: int):
    env_id = ENV_IDS[env_key]
    env = gym.make(env_id)
    # Seed everything reproducibly
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    obs, info = env.reset(seed=seed)
    return env, obs

def main(args):
    cfg = load_config("config.json")
    env, _ = make_env(args.env, args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(cfg.get("hidden_dims", [64, 64])),
        activation_fn=get_activation_fn(cfg.get("activation_fn", "tanh")),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        policy_lr=cfg.get("policy_lr", 3e-4),
        value_lr=cfg.get("value_lr", 3e-4),
        gamma=cfg.get("gamma", 0.99),
        vf_coef=cfg.get("vf_coef", 0.5),
        ent_coef=cfg.get("ent_coef", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        use_gae=args.use_gae,
        gae_lambda=cfg.get("gae_lambda", 0.95),
    )

    episode_rewards = []
    moving_avg = []

    for ep in range(1, args.episodes + 1):
        ep_reward = 0.0
        steps = 0
        state, _ = env.reset(seed=args.seed + ep)  # vary seed per episode
        done = False
        while not done and steps < args.max_timesteps:
            action, _ = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            agent.step((state, action, float(reward), next_state, done))
            state = next_state
            ep_reward += float(reward)
            steps += 1

        episode_rewards.append(ep_reward)
        avg = np.mean(episode_rewards[-10:])
        moving_avg.append(avg)
        print(f"Episode {ep:4d} | reward={ep_reward:8.2f} | 10-ep moving avg={avg:8.2f} | steps={steps}")

    # Plot (optional; safe in headless)
    plt.figure()
    plt.plot(episode_rewards, label="Episode reward")
    plt.plot(moving_avg, label="Moving avg (10)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    if args.save_path:
        # Save only policy weights (matches your test loader)
        torch.save(agent.policy.state_dict(), args.save_path)
        print(f"Saved policy weights to {args.save_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_IDS.keys()), default="mountaincar")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_timesteps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="a2c_model.pth")
    parser.add_argument("--use_gae", action="store_true")
    args = parser.parse_args()
    main(args)
