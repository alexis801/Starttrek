##
## EPITECH PROJECT, 2026
## startrek
## File description:
## eval.py
##

import argparse
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn

from bad_model.basic import basic_policy
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_merged_config
from utils.seed import set_seed

CONFIG_DEFAULT = os.path.join(PROJECT_ROOT, "configs/default.yaml")
CONFIG_EVAL = os.path.join(PROJECT_ROOT, "configs/eval.yaml")

DEFAULT_VIDEO_DIR = "stats/videos"

def resolve_model_path(model_path: str) -> Optional[Path]:
    requested_path = Path(model_path)
    if requested_path.exists():
        return requested_path

    fallback_paths = [
        Path("models/model_dqn.pth"),
        Path("models/model_double_dqn.pth"),
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
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    # Load default.yaml + eval.yaml (eval overrides default)
    cfg = load_merged_config(CONFIG_DEFAULT, CONFIG_EVAL)

    # Apply CLI overrides (CLI > YAML
    if args.model is not None:
        cfg["eval"]["model_path"] = args.model
    if args.episodes is not None:
        cfg["eval"]["num_episodes"] = args.episodes
    if args.csv_out is not None:
        cfg["eval"]["csv_output"] = args.csv_out
    if args.render_mode is not None:
        cfg["env"]["render_mode"] = args.render_mode
    if args.seed is not None:
        cfg["seeds"]["base"] = args.seed
    return cfg


def create_env(env_name: str, render_mode: str, record_video: bool, video_dir: str) -> gym.Env:
    gym_render_mode = "rgb_array" if record_video else render_mode
    env = gym.make(env_name, render_mode=gym_render_mode)
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


def select_action(obs: Any, basic: bool, model: Optional[Any]) -> int:
    if basic:
        return basic_policy(obs)
    if model is None:
        raise ValueError("DQN policy selected but no model found.")
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def run_episode(env: gym.Env, seed: int, basic: bool, model: Optional[Any]) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = select_action(obs, basic, model)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

    return EpisodeResult(
        reward=total_reward,
        terminated=terminated,
        truncated=truncated,
    )


def evaluate_policy(env: gym.Env, num_episodes: int, base_seed: int,
    basic: bool, model: Optional[Any]) -> list[EpisodeResult]:
    results = []
    for episode_idx in range(num_episodes):
        results.append(run_episode(env, base_seed + episode_idx, basic, model))
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
    cfg = resolve_config(args)

    # read final values from merged config
    num_episodes = cfg["eval"]["num_episodes"]
    model_path = cfg["eval"]["model_path"]
    csv_output = cfg["eval"]["csv_output"]
    env_name = cfg["env"]["name"]
    render_mode = cfg["env"]["render_mode"]
    base_seed = cfg["seeds"]["base"]

    if num_episodes <= 0:
        raise ValueError("eval.num_episodes must be greater than 0")
    if args.record_video and render_mode == "human":
        print("Note: --record-video forces render_mode to rgb_array (no live window).")

    # reproducibility
    set_seed(base_seed)

    env = create_env(env_name, render_mode, args.record_video, args.video_dir)
    try:
        env.action_space.seed(base_seed)
        model = None
        if not args.basic:
            model = load_model(model_path)
        results = evaluate_policy(env, num_episodes, base_seed, args.basic, model)
        print_episode_results(results)
        print_summary(results)
        save_results_to_csv(results, csv_output)
        if args.record_video:
            print(f"Videos saved under '{Path(args.video_dir).resolve()}'.")
    finally:
        env.close()

if __name__ == "__main__":
    main()