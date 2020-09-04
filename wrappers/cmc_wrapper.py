import gym


class MountainCarContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the MountainCarContinuous environment
    """
    def __init__(self, env):
        super(MountainCarContinuousWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        return next_state, 0.1 + reward, done, y
