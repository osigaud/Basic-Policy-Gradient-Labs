import gym


class PendulumWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the pendulum environment
    """
    def __init__(self, env):
        super(PendulumWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        return next_state, (8+reward)/10, done, y