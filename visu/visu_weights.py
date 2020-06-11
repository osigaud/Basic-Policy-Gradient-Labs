import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from policies import GenericNet, PolicyWrapper
from visu.visu_policies import final_show


def random_state():
    x1 = random.random() * 4.8 - 2.4
    x2 = random.random() * 4.8 - 2.4
    dx1 = random.random() - 0.5
    dx2 = random.random() - 0.5
    return np.array((x1, dx1, x2, dx2))


def random_state_vector():
    retour = []
    for i in range(2000):
        retour.append(random_state())
    return retour


def get_weight_sample(net):
    retour = []
    states = random_state_vector()
    for st in states:
        probs = net.forward(st)
        action = probs.item()
        retour.append(action)
    return retour


def get_normal_sample(net):
    mus = []
    stds = []
    states = random_state_vector()
    for st in states:
        mu, std = net.forward(st)
        mu = mu.data.numpy().astype(float)
        std = std.data.numpy().astype(float)
        mus.append(mu)
        stds.append(std)
    return mus, stds

def plot_normal_histograms(network):
    mus, stds = get_normal_sample(network)
    mus = np.array(mus)
    stds = np.array(stds)
    plt.figure(1, figsize=(13, 10))

    bar_width = 0.0005
    bins_mus = np.arange(mus.min(), mus.max() + bar_width, bar_width)
    bins_stds = np.arange(stds.min(), stds.max() + bar_width, bar_width)
    plt.hist(mus, bins=bins_mus)
    final_show(True, True, 'dispersion_mu.pdf', "mu", "count", "dispersion mu", '/results/')

    plt.hist(stds, bins=bins_stds)
    final_show(True, True, 'dispersion_std.pdf', "variance", "count", "dispersion variance", '/results/')

def plot_weight_histogram(network):
    probas = np.array(get_weight_sample(network))
    plt.figure(1, figsize=(13, 10))

    bar_width = 0.0005
    bins = np.arange(probas.min(), probas.max() + bar_width, bar_width)
    plt.hist(probas, bins=bins)
    final_show(True, True, 'dispersion.pdf', "decision threshold", "count", "decision dispersion", '/results/')


if __name__ == '__main__':
    pdirectory = os.getcwd() + '/policies/'
    pw = PolicyWrapper(GenericNet(), "", "")
    #policy = pw.load(pdirectory + "CartPole-v0#top1_198.6#198.6.pt")
    policy = pw.load(pdirectory + "CartPoleContinuous-v0#moi_30.55#30.55.pt")
    # policy = PolicyNetNormal(4, 24, 36, 1, 0.01)
    plot_normal_histograms(policy)
