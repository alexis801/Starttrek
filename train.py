import gymnasium as gym
import torch

def policy(obs, env):
    return env.action_space.sample()

def episode_loop(env, episode_over, total_reward, obs):
    while not episode_over:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
    print(f"Episode finished! Total reward: {total_reward}")

def episode_handling(env, obs):
    episode_over = False
    total_reward = 0

    episode_loop(env, episode_over, total_reward, obs)

def main():
    SEED = 42
    env = gym.make("LunarLander-v3", render_mode="human")

    obs, info = env.reset(seed=SEED)
    print(f"Starting observation: {obs}")

    episode_handling(env, obs)
    env.close()
    # torch.save(env.state_dict(), "model.pth")

if __name__ == "__main__":
    main()