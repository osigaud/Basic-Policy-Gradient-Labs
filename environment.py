import gym
import my_gym
import numpy as np
from wrappers import FeatureInverter, BinaryShifter, ActionAdapter, PerfWriter

# pour voir la liste des environnements gym disponible
# from gym import envs
# print(envs.registry.all())

def make_env(env_name, policy_type, env_obs_space_name=None):
    if env_name == "CartPole-v0" or env_name == "CartPoleContinuous-v0":
        env = FeatureInverter(gym.make(env_name), 1, 2)
        if env_name == "CartPoleContinuous-v0":
            if policy_type == "bernoulli":
                env = BinaryShifter(env)
        env.observation_space.names = env_obs_space_name
    else:
        env = gym.make(env_name)
    # a way to know whether the environment takes discrete or continuous actions
    # will only work for simple 1D action classic control benchmarks
    discrete = not env.action_space.contains(np.array([0.5]))
    if discrete:
        env = ActionAdapter(env)
    env = PerfWriter(env)
    return env, discrete
