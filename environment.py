import gym
from gym.wrappers import TimeLimit
import numpy as np

from wrappers import FeatureInverter, PerfWriter


def make_env(env_name, policy_type, max_episode_steps, env_obs_space_name=None):
    """
    Wrap the environment into a set of wrappers depending on some hyper-parameters
    Used so that most environments can be used with the same policies and algorithms
    :param env_name: the name of the environment, as a string. For instance, "MountainCarContinuous-v0"
    :param policy_type: a string specifying the type of policy. So far, "bernoulli" or "normal"
    :param max_episode_steps: the max duration of an episode. If None, uses the default gym max duration
    :param env_obs_space_name: a vector of names of the environment features. E.g. ["position","velocity"] for
    MountainCar
    :return: the wrapped environment
    """
    env = gym.make(env_name)
    # tests whether the environment is discrete or continuous
    if not env.action_space.contains(np.array([0.5])):
        assert policy_type == "bernoulli" or policy_type == "discrete", 'cannot run a continuous action policy in a ' \
                                                                        'discrete action environment'

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    env.observation_space.names = env_obs_space_name

    env = FeatureInverter(env, 1, 2)  # MODIFIED: Invert sin(theta) and theta dot

    env = PerfWriter(env)
    print(env)
    return env

# to see the list of available gym environments, type:
# from gym import envs
# print(envs.registry.all())
