from tqdm import tqdm
import numpy as np

class QLAgent():
    def __init__(self,state_space_size, action_space_size):
        self.rng = np.random.default_rng()
        self.qtable = np.zeros((state_space_size, action_space_size))

    def __greedy_policy(self, state):
        action = np.argmax(self.qtable[state])
        return action

    def __epsilon_greedy_policy(self, env, state, epsilon):
        random_num = self.rng.uniform(0,1)
        if random_num > epsilon: 
            action = self.__greedy_policy(state)
        else:
            action = env.action_space.sample()

        return action

    def train(self, n_training_episodes, env, max_steps, learning_rate, discount_factor=0.99, max_epsilon=1.0, min_epsilon=0.0, decay=1e-5):
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay*episode)
            state, _ = env.reset()

            for step in range(max_steps):
                action = self.__epsilon_greedy_policy(env, state, epsilon)
                new_state, reward, terminated, truncated, info = env.step(action)
                
                current_value = self.qtable[state][action]
                max_util_next_move = np.max(self.qtable[new_state])
                self.qtable[state][action] = current_value + learning_rate * (
                    reward + discount_factor * max_util_next_move - current_value
                )

                if terminated or truncated: break
                state = new_state


    def evaluate(self, env, n_eval_episodes, max_steps, seeds = None):
        episode_rewards = []
        for episode in tqdm(range(n_eval_episodes)):
            if seeds is not None:
                state, *_ = env.reset(seed=seeds[episode])
            else: 
                state, *_ = env.reset()
            step = 0
            total_rewards_ep = 0

            for step in range(max_steps):
                action = self.__greedy_policy(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                total_rewards_ep += reward

                if terminated or truncated: break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        
        mean_reward = np.mean(episode_rewards)
        std_reward  = np.std(episode_rewards)

        return mean_reward, std_reward

    def act(self, state):
        action_taken = self.__greedy_policy(state)
        return action_taken