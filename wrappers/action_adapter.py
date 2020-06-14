import gym
from gym import Wrapper
from gym import version

class ActionAdapter(Wrapper):
    """
    Some gym environment take a scalar as action input for the step function, others take a vector of size 1
    This wrapper is used so that all environment can be used uniformly as taking a vector of size 1:
    It will take the scalar content of the first cell of the vector as the input of step for those which take a scalar
    """
    def __init__(self, env):
        super(ActionAdapter, self).__init__(env)

    def _step(self, action):
        act = action[0]
        return self.env.step(act)
 
    def _env_info(self):
        env_info = {
            'gym_version': version.VERSION,
        }
        if self.env.spec:
            env_info['env_id'] = self.env.spec.id
        return env_info
