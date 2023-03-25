import gym
import numpy as np
class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
   
    def policy(self, obs, theta):
        action_prob = np.dot(theta, obs)
        action = 1 if action_prob > 0 else 0
        return action

    def cross_entropy_method(self, n_iterations, n_samples, percentile):
        theta = np.zeros(self.env.observation_space.shape[0])
        sigma = 1.0

        for i in range(n_iterations):
            policies = [np.random.normal(theta, sigma, theta.shape) for _ in range(n_samples)]
            rewards = []
            for policy_params in policies:
                obs = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = self.policy(obs, policy_params)
                    obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                rewards.append(total_reward)

            elite_idx = np.argsort(rewards)[-int(percentile*n_samples):]
            elite_policies = [policies[i] for i in elite_idx]

            theta = np.mean(elite_policies, axis=0)

            sigma = np.std(elite_policies, axis=0)

            print(f"Average reward for iteration {i+1}: {np.mean(rewards)}")

cp = CartPole()
cp.cross_entropy_method(n_iterations=50, n_samples=100, percentile=0.5)
