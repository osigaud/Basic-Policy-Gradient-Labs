import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this
import numpy as np
from wrappers import FeatureInverter, BinaryShifter, ContinuousActionAdapter, BinaryActionAdapter, PerfWriter
from gym.wrappers import TimeLimit

# to see the list of available gym environments, type:
# from gym import envs
# print(envs.registry.all())

def make_env(env_name, policy_type, env_obs_space_name=None):
    if env_name == "CartPole-v0" or env_name == "CartPoleContinuous-v0":
        env = FeatureInverter(gym.make(env_name), 1, 2)
        if env_name == "CartPoleContinuous-v0":
            if policy_type == "bernoulli":
                env = BinaryShifter(env)
            else:
                env = ContinuousActionAdapter(env)
            env = TimeLimit(env, 200)
        env.observation_space.names = env_obs_space_name
    else:
        env = gym.make(env_name)
    # a way to know whether the environment takes discrete or continuous actions
    # will only work for simple 1D action classic control benchmarks
    discrete = not env.action_space.contains(np.array([0.5]))
    if discrete:
        env = BinaryActionAdapter(env)
    env = PerfWriter(env)
    return env, discrete
