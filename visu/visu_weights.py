import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from policies import GenericNet, PolicyWrapper
from visu.visu_policies import final_show

# The two functions below are not generic
def random_cartpole_state():
    """
    Return a random state from the CartPole (or CartPoleContinuous) environment
    :return:  the random state
    """
    x1 = random.random() * 4.8 - 2.4
    x2 = random.random() * 4.8 - 2.4
    dx1 = random.random() - 0.5
    dx2 = random.random() - 0.5
    return np.array((x1, dx1, x2, dx2))


def random_mountaincar_state():
    """
    Return a random state from the MountainCar (or MountainCarContinuous) environment
    :return:  the random state
    """
    x1 = random.random() * 2 - 1
    x2 = random.random() * 4.8 - 2.4
    return np.array((x1, x2))


def random_state_vector(env_name):
    """
    Return a set of random states
    :param env_name:
    :return: a vector of random states
    """
    assert env_name in ['CartPoleContinuous-v0', 'MountainCarContinuous-v0'], 'unsupported environment'
    random_states = []
    for i in range(2000):
        if env_name == 'CartPoleContinuous-v0':
            random_states.append(random_cartpole_state())
        else:
            random_states.append(random_mountaincar_state())
    return random_states


def get_weight_sample(policy, env_name):
    """
    Return the sample of output weights of a Bernoulli policy obtained from a vector of random states
    :param policy: the policy network
    :param env_name: the name of the environment
    :return: a vector of output weights
    """
    weights = []
    states = random_state_vector(env_name)
    for st in states:
        probs = policy.forward(st)
        action = probs.item()
        weights.append(action)
    return weights


def get_normal_sample(policy, env_name):
    """
    Return the sample of Gaussian parameters of a Gaussian policy obtained from a vector of random states
    :param policy: the policy network
    :param env_name: the name of the environment
    :return: a vector of Gaussian parameters
    """
    mus = []
    stds = []
    states = random_state_vector(env_name)
    for st in states:
        mu, std = policy.forward(st)
        mu = mu.data.numpy().astype(float)
        mus.append(mu)
        stds.append(std)
    return mus, stds

def plot_normal_histograms(policy, nb, env_name) -> None:
    """
    
    :param policy: the policy network
    :param nb: a number to allow several such plots through repeated epochs
    :param env_name: the name of the environment
    :return: nothing
    """
    mus, stds = get_normal_sample(policy, env_name)
    mus = np.array(mus)
    stds = np.array(stds)
    plt.figure(1, figsize=(13, 10))

    bar_width = 0.0005
    bins_mus = np.arange(mus.min(), mus.max() + bar_width, bar_width)
    bins_stds = np.arange(stds.min(), stds.max() + bar_width, bar_width)
    plt.hist(mus, bins=bins_mus)
    final_show(True, False, 'dispersion_mu_' + str(nb) + '.pdf', "mu", "count", "dispersion mu", '/results/')

    plt.hist(stds, bins=bins_stds)
    final_show(True, False, 'dispersion_std_' + str(nb) + '.pdf', "variance", "count", "dispersion variance", '/results/')

def plot_weight_histograms(policy, nb, env_name) -> None:
    """
    :param policy: the policy network
    :param nb: a number to allow several such plots through repeated epochs
    :param env_name: the name of the environment
    :return: nothing
    """
    probas = np.array(get_weight_sample(policy, env_name))
    plt.figure(1, figsize=(13, 10))

    bar_width = 0.0005
    bins = np.arange(probas.min(), probas.max() + bar_width, bar_width)
    plt.hist(probas, bins=bins)
    final_show(True, False, 'dispersion_' + str(nb) + '.pdf', "decision threshold", "count", "decision dispersion", '/results/')
