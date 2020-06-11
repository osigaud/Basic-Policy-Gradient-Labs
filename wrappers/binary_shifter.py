import gym
from gym import Wrapper
from gym import version


class BinaryShifter(Wrapper):
    """
    This wrapper is used to transform the {0,1} output of a binary policy into the {-1,1} action
    space that most gym environment are expecting
    The input action given to state is assumed to be a vector of size 1, hence we get it from action[0],
    see the ActionAdapter wrapper
    """
    def __init__(self, env):
        super(BinaryShifter, self).__init__(env)

    def _step(self, action):
        if action[0] == 0.0:
            act = -1.0
        else:
            act = action[0]
        if not (act == 1.0 or act == -1.0):
            print ("binary shifter : action = ", action[0])
        return self.env.step(act)
 
    def _env_info(self):
        env_info = {
            'gym_version': version.VERSION,
        }
        if self.env.spec:
            env_info['env_id'] = self.env.spec.id
        return env_info
