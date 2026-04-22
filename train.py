import gymnasium as gym
import torch
SEED = 42

env = gym.make("LunarLander-v3", render_mode="human")

def policy(obs):
    return env.action_space.sample()

obs, info = env.reset(seed=SEED)


print(f"Starting observation: {obs}")

episode_over = False
total_reward = 0

while not episode_over:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")

env.close()

torch.save(env.state_dict(), "model.pth")