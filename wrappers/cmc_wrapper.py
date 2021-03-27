import gym


class MountainCarContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the MountainCarContinuous environment
    """
    def __init__(self, env, shift = 0):
        super(MountainCarContinuousWrapper, self).__init__(env)
        self.reward_shift = shift

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        return next_state, self.reward_shift + reward, done, y
        # return next_state, reward, done, y
