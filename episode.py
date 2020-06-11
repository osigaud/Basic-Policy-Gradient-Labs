import numpy as np
import math


class Episode:
    def __init__(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.done_pool = []
        self.next_state_pool = []
        self.len = 0

    def add(self, state, action, reward, done, next_state):
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.done_pool.append(done)
        self.next_state_pool.append(next_state)
        self.len += 1

    def size(self):
        return self.len

    def discounted_sum_rewards(self, gamma):
        summ = 0
        for i in reversed(range(len(self.state_pool))):
            summ = summ * gamma + self.reward_pool[i]
            self.reward_pool[i] = summ

    def sum_rewards(self):
        summ = np.sum(self.reward_pool)
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = summ

    def substract_baseline(self, critic):
        val = critic.evaluate(np.array(self.state_pool), np.array(self.action_pool))
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] -= val[i][0]

    # attention, on réécrit sur les reward_pool => à faire dans le bon sens
    def nstep_return(self, n, gamma, critic):
        for i in range(len(self.reward_pool)):
            horizon = i + n
            summ = self.reward_pool[i]
            if horizon < len(self.reward_pool):
                bootstrap_val = critic.evaluate(self.state_pool[horizon], self.action_pool[horizon])[0]
                summ += gamma ** n * bootstrap_val
            for j in range(1, n):
                if i + j >= len(self.reward_pool):
                    break
                summ += gamma**j * self.reward_pool[i+j]
            self.reward_pool[i] = summ

    def normalize_rewards(self, reward_mean, reward_std):
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

    def normalize_discounted_rewards(self, gamma, reward_mean, reward_std):
        summ = 0
        for i in reversed(range(len(self.state_pool))):
            summ = summ * gamma + self.reward_pool[i]
            self.reward_pool[i] = (summ - reward_mean) / reward_std

    def exponentiate_rewards(self, beta):
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = math.exp(self.reward_pool[i]/beta)