import gym


class BinaryShifterDiscrete(gym.Wrapper):
    """
    This wrapper is used to transform the {0,1} output of a binary policy
    into the {-1,1} action space that most gym environment are expecting
    The input action given to state is assumed to be a vector of size 1, hence we get it from action[0],
    see the ActionAdapter wrapper
    """
    def __init__(self, env):
        super(BinaryShifterDiscrete, self).__init__(env)

    def step(self, action):
        assert (action[0] == 1.0 or action[0] == 0.0), 'binary shifter : action =' + action[0]
        if action[0] == 0.0:
            return self.env.step([0])
        else:
            return self.env.step([1])

