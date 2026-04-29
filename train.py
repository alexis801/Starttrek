import gymnasium as gym
import torch
import torch.nn as nn
import random

epsilon = 0.1

def policy(obs, env):
    if random.random() < epsilon:
        return env.action_space.sample()

    x, y, vx, vy, angle, v_angle, left, right = obs

    if angle > 0.1:
        return 3
    elif angle < -0.1:
        return 1
    elif vy < -0.5:
        return 2
    else:
        return 0

def episode_loop(env, episode_over, total_reward, obs):
    while not episode_over:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
    print(f"Episode finished! Total reward: {total_reward}")

def episode_handling(env, obs, rewards_log):
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    rewards_log.append(total_reward)

def main():
    SEED = 42
    env = gym.make("LunarLander-v3")

    rewards_log = []

    for episode in range(100):
        obs, info = env.reset()
        episode_handling(env, obs, rewards_log)

    print(sum(rewards_log) / len(rewards_log))

    env.close()
    # torch.save(env.state_dict(), "model.pth")

if __name__ == "__main__":
    main()