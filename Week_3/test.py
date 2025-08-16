import argparse
import json
import gymnasium as gym
import numpy as np
import torch
from assets import A2C, device

ENV_IDS = {
    "mountaincar": "MountainCarContinuous-v0",
    "hopper": "Hopper-v4",
}

def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)

def evaluate(args):
    cfg = load_config("config.json")
    env = gym.make(ENV_IDS[args.env])
    _obs, _info = env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    act_fn = torch.tanh if cfg.get("activation_fn", "tanh").lower() == "tanh" else torch.relu

    agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(cfg.get("hidden_dims", [64, 64])),
        activation_fn=act_fn,
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


    # Load saved policy weights
    model_path = args.model_path or "a2c_model.pth"
    sd = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(sd)
    agent.policy.eval()

    returns = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < args.max_timesteps:
            # Deterministic action for evaluation (mean of the policy)
            action, _ = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state = next_state
            total += float(reward)
            steps += 1
        returns.append(total)
        print(f"[Eval] Episode {ep+1}/{args.episodes} return = {total:.2f} steps={steps}")

    print(f"Average return over {args.episodes} episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_IDS.keys()), default="hopper",
                        help="Which environment to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--max_timesteps", type=int, default=1500, help="Max timesteps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model (defaults to a2c_model.pth)")
    parser.add_argument("--use_gae", action="store_true", help="Use GAE (not needed for eval)")
    args = parser.parse_args()
    evaluate(args)
