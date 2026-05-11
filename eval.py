##
## EPITECH PROJECT, 2026
## startrek
## File description:
## eval.py
##

import argparse
import csv
import gymnasium as gym
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from basic import *

DEFAULT_SEED = 42
DEFAULT_NUM_EPISODES = 5
DEFAULT_MODEL_PATH = "model.pth"
DEFAULT_RENDER_MODE = "rgb_array"
DEFAULT_CSV_OUTPUT = "stats/csv/eval_results.csv"
DEFAULT_VIDEO_DIR = "stats/videos"

def resolve_model_path(model_path: str) -> Optional[Path]:
    requested_path = Path(model_path)
    if requested_path.exists():
        return requested_path

    if model_path == DEFAULT_MODEL_PATH:
        fallback_paths = [
            Path("models") / "model.pth",
            Path("modele.pth"),
            Path("modèle.pth"),
            Path("models") / "modele.pth",
            Path("models") / "modèle.pth",
        ]
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                print(f"Using fallback model path '{fallback_path}'.")
                return fallback_path

    return None

class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class EpisodeResult:
    reward: float
    terminated: bool
    truncated: bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LunarLander-v3 policy."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--render-mode", default=DEFAULT_RENDER_MODE)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--policy", choices=["basic", "dqn"], default="dqn",
)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save one MP4 per episode under --video-dir (requires rgb_array).",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        help="Directory for RecordVideo output (created if missing).",
    )
    return parser.parse_args()


def create_env(
    render_mode: str,
    record_video: bool,
    video_dir: str,
) -> gym.Env:
    gym_render_mode = "rgb_array" if record_video else render_mode
    env = gym.make("LunarLander-v3", render_mode=gym_render_mode)
    if record_video:
        out = Path(video_dir)
        out.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            str(out),
            episode_trigger=lambda episode_id: True,
            name_prefix="lunar_lander",
            disable_logger=True,
        )
    return env

def load_model(model_path: str) -> Optional[Any]:
    path = resolve_model_path(model_path)
    if path is None:
        print(f"No model found at '{model_path}', using random policy.")
        return None

    loaded_object = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(loaded_object, nn.Module):
        loaded_object.eval()
        return loaded_object

    if isinstance(loaded_object, dict):
        model = QNetwork()
        model.load_state_dict(loaded_object)
        model.eval()
        return model

    raise TypeError(
        f"Unsupported model format at '{model_path}': {type(loaded_object).__name__}"
    )

def select_action(
    env: gym.Env, obs: Any, policy_type: str, model: Optional[Any],) -> int:

    if policy_type == "basic":
        return basic_policy(obs)
    if model is None:
        raise ValueError("DQN policy selected but no model find")
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_tensor)
    return int(torch.argmax(q_values, dim=1).item())

def run_episode(env: gym.Env, seed: int, policy_type: str, model: Optional[Any],) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = select_action(env, obs, policy_type, model)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

    return EpisodeResult(
        reward=total_reward,
        terminated=terminated,
        truncated=truncated,
    )

def evaluate_policy(env: gym.Env, num_episodes: int, base_seed: int, policy_type: str, model: Optional[Any],) -> list[EpisodeResult]:
    results = []
    for episode_idx in range(num_episodes):
        results.append(run_episode(env, base_seed + episode_idx, policy_type, model))
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
    terminated_count = sum(1 for result in results if result.terminated)
    truncated_count = sum(1 for result in results if result.truncated)
    min_reward = min(result.reward for result in results)
    max_reward = max(result.reward for result in results)
    print(f"Mean reward over {len(results)} episodes: {mean_reward:.2f}")
    print(
        "Termination stats: "
        f"terminated={terminated_count} truncated={truncated_count}"
    )
    print(f"Reward range: min={min_reward:.2f} max={max_reward:.2f}")


def save_results_to_csv(results: list[EpisodeResult], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward", "terminated", "truncated"])
        for episode_idx, result in enumerate(results, start=1):
            writer.writerow(
                [
                    episode_idx,
                    f"{result.reward:.6f}",
                    int(result.terminated),
                    int(result.truncated),
                ]
            )
    print(f"Saved episode results to '{output_file}'.")

def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.record_video and args.render_mode == "human":
        print(
            "Note: --record-video forces render_mode to rgb_array (no live window)."
        )
    env = create_env(args.render_mode, args.record_video, args.video_dir)
    try:
        model = None
        if args.policy == "dqn":
            model = load_model(args.model)
        results = evaluate_policy(env, args.episodes, args.seed, args.policy, model)
        print_episode_results(results)
        print_summary(results)
        save_results_to_csv(results, args.csv_out)
        if args.record_video:
            print(f"Videos saved under '{Path(args.video_dir).resolve()}'.")
    finally:
        env.close()

if __name__ == "__main__":
    main()