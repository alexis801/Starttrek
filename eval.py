##
## EPITECH PROJECT, 2026
## startrek
## File description:
## eval.py
##

import gymnasium as gym
from dataclasses import dataclass

SEED = 42
NUM_EPISODES = 5

@dataclass
class EpisodeResult:
    reward: float
    terminated: bool
    truncated: bool

def create_env() -> gym.Env:
    return gym.make("LunarLander-v3", render_mode="rgb_array")

def run_episode(env: gym.Env, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    return EpisodeResult(
        reward=total_reward,
        terminated=terminated,
        truncated=truncated,
    )

def evaluate_random_policy(
    env: gym.Env,
    num_episodes: int,
    base_seed: int,
) -> list[EpisodeResult]:
    results = []
    for episode_idx in range(num_episodes):
        results.append(run_episode(env, base_seed + episode_idx))
    return results

def print_episode_results(results: list[EpisodeResult]) -> None:
    for episode_idx, result in enumerate(results, start=1):
        print(
            f"[Episode {episode_idx}/{len(results)}] "
            f"reward={result.reward:.2f} "
            f"terminated={result.terminated} "
            f"truncated={result.truncated}"
        )

def print_summary(results: list[EpisodeResult]) -> None:
    mean_reward = sum(result.reward for result in results) / len(results)
    print(f"Mean reward over {len(results)} episodes: {mean_reward:.2f}")

def main() -> None:
    env = create_env()
    results = evaluate_random_policy(env, NUM_EPISODES, SEED)
    print_episode_results(results)
    print_summary(results)
    env.close()

if __name__ == "__main__":
    main()