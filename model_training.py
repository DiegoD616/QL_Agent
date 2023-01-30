import gymnasium as gym
import random
import imageio
from QL_Agent import QLAgent as Agent
import numpy as np

def main():
    env_params = {"id":"FrozenLake-v1", "is_slippery":False, "map_name":"8x8"}
    train_env  = gym.make(**env_params)
    state_space_size  = train_env.observation_space.n
    action_space_size = train_env.action_space.n
    agent             = Agent(state_space_size, action_space_size)

    TRAINING_HYPERPARAMETERS = {
        "env": train_env,
        "n_training_episodes" : 5000, # Total training episodes
        "learning_rate" : 0.75,       # Learning rate
        "max_steps" : int(1e3),       # Max steps per episode
        "discount_factor" : 0.98,     # Discounting rate
        "max_epsilon" : 1.0,          # Exploration probability at start
        "min_epsilon" : 0.3,          # Minimum exploration probability
        "decay" : 0.00005,            # Exponential decay rate for exploration prob
    }
    agent.train(**TRAINING_HYPERPARAMETERS)

    EVAL_PARAMETERS = {
        "env": gym.make(**env_params),
        "n_eval_episodes" : 100,  # Total number of test episodes
        "max_steps" : int(1e3),   # Max steps per episode
    }
    mean_reward, std_reward = agent.evaluate(**EVAL_PARAMETERS)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    record_video(gym.make(render_mode="rgb_array",**env_params), agent)

def record_video(env, agent, out_directory="./replay.mp4", fps=1):
    images = []
    terminated, truncated = False, False
    state, _ = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated and not truncated:
        action = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

if __name__ == "__main__":
    main()
