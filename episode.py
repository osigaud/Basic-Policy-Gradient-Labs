import numpy as np
import math


class Episode:
    """
    This class stores the samples of an episode
    """
    def __init__(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.done_pool = []
        self.next_state_pool = []
        self.len = 0

    def add(self, state, action, reward, done, next_state) -> None:
        """
        Add a sample to the episode
         :param state: the current state
        :param action: the taken action
        :param reward: the resulting reward
        :param done: whether the episode is over
        :param next_state: the resulting next state
        :return: nothing
        """
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.done_pool.append(done)
        self.next_state_pool.append(next_state)
        self.len += 1

    def size(self):
        """
        Return the number of samples already stored in the episode
        :return: the number of samples in the episode
        """
        return self.len

    def discounted_sum_rewards(self, gamma) -> None:
        """
        Apply a discounted sum of rewards to all samples of the episode
        :param gamma: the discount factor
        :return: nothing
        """
        summ = 0
        for i in reversed(range(len(self.state_pool))):
            summ = summ * gamma + self.reward_pool[i]
            self.reward_pool[i] = summ

    def sum_rewards(self) -> None:
        """
        Apply a sum of rewards to all samples of the episode
        :return: nothing
        """
        summ = np.sum(self.reward_pool)
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = summ

    def substract_baseline(self, critic) -> None:
        """
        Substracts a baseline to the reward of all samples of the episode
        :param critic: the baseline critic to be substracted
        :return: nothing
        """
        val = critic.evaluate(np.array(self.state_pool), np.array(self.action_pool))
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] -= val[i][0]

    def nstep_return(self, n, gamma, critic) -> None:
        """
         Apply Bellman backup n-step return to all rewards of all samples of the episode
         Warning, we rewrite on reward_pools, must be done in the right order
         :param n: the number of steps in n-step
         :param gamma: the discount factor
         :param critic: the critic used to perform Bellman backups
         :return: nothing
         """
        for i in range(len(self.reward_pool)):
            horizon = i + n
            summ = self.reward_pool[i]
            if horizon < len(self.reward_pool):
                bootstrap_val = critic.evaluate(self.state_pool[horizon], self.action_pool[horizon])[0][0]
                summ += gamma ** n * bootstrap_val
            for j in range(1, n):
                if i + j >= len(self.reward_pool):
                    break
                summ += gamma**j * self.reward_pool[i+j]
            self.reward_pool[i] = summ

    def normalize_rewards(self, reward_mean, reward_std) -> None:
        """
        Apply a normalized sum of rewards (non discounted) to all samples of the episode
        :param gamma: the discount factor
        :return: nothing
        """
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

    def normalize_discounted_rewards(self, gamma, reward_mean, reward_std) -> None:
        """
         Apply a normalized and discounted sum of rewards to all samples of the episode
         :param gamma: the discount factor
         :return: nothing
         """
        summ = 0
        for i in reversed(range(len(self.state_pool))):
            summ = summ * gamma + self.reward_pool[i]
            self.reward_pool[i] = (summ - reward_mean) / reward_std

    def exponentiate_rewards(self, beta) -> None:
        """
          Apply an exponentiation factor to the rewards of all samples of the episode
          :param beta: the exponentiation factor
          :return: nothing
          """
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = math.exp(self.reward_pool[i]/beta)