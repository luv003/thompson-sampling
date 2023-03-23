import numpy as np 
import random 
import math 
import matplotlib.pyplot as plt 


def bernoulli(p):
    if random.random()<p:
        return 1
    else:
        return 0 

def thompson_sampling(num_arms,num_iterations,true_means):
    sucesses=np.zeros(num_arms)
    failures=np.zeros(num_arms)
    samples=np.zeros(num_arms)
    arm_values=np.zeros(num_arms)
    total_reward=0 
    reward_history=[]


    for i in range(num_iterations):
        for j in range(num_arms):
            arm_values[j]=np.random.beta(sucesses[j]+1,failures[j]+1)
            chosen_arm=np.argmax(arm_values)
            reward=bernoulli(true_means[chosen_arm])

            if reward==1:
                sucesses[chosen_arm]+=1
            else:
                failures[chosen_arm]+=1
            samples[chosen_arm]+=1
            total_reward+=reward
            reward_history.append(total_reward)
        success_rates=sucesses/samples
        return reward_history,success_rates







num_arms=50
true_means=np.random.uniform(0,1,num_arms)
num_iterations=1000 
reward_history, success_rates = thompson_sampling(num_arms, num_iterations, true_means)

plt.plot(reward_history)
plt.xlabel('Iteration')
plt.ylabel('Total Reward')
plt.title('Thompson Sampling')
plt.show()

plt.bar(np.arange(num_arms), success_rates)
plt.xlabel('Arm')
plt.ylabel('Success Rate')
plt.title('Thompson Sampling Success Rates')
plt.show()
































































# import numpy as np

# class ThompsonSampling:
#     def __init__(self, n_arms, alpha, beta):
#         self.n_arms = n_arms
#         self.alpha = alpha
#         self.beta = beta
#         self.reward_sum = np.zeros(n_arms)
#         self.trial_sum = np.zeros(n_arms)

#     def select_arm(self):
#         theta = np.zeros(self.n_arms)
#         for i in range(self.n_arms):
#             theta[i] = np.random.beta(self.alpha + self.reward_sum[i], self.beta + self.trial_sum[i] - self.reward_sum[i])
#         return np.argmax(theta)

#     def update(self, arm, reward):
#         self.reward_sum[arm] += reward
#         self.trial_sum[arm] += 1
