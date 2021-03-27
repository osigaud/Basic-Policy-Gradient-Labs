import gym
from math import pi

class PendulumWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the pendulum environment
    """
    def __init__(self, env, shift):
        super(PendulumWrapper, self).__init__(env)
        self.min_reward_ = -(pi ** 2 + 0.1 * 8 ** 2 + 0.001 * 2 ** 2)
        self.reward_shift = shift

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        return next_state, self.reward_shift + reward, done, y
