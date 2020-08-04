import gym


class BinaryActionAdapter(gym.Wrapper):
    """
    This wrapper is used so that environments which take 0 or 1 as action can be used as taking a vector of size 1:
    """
    def __init__(self, env):
        super(BinaryActionAdapter, self).__init__(env)

    def step(self, action):
        act = action[0]
        if act==1.0:
            act = 1
        else:
            act = 0
        return self.env.step(act)