import numpy as np
import matplotlib.pyplot as plt
from visu.visu_policies import final_show

def episode_to_traj(episode):
    x = []
    y = []
    for state in episode.state_pool:
        x.append(state[0])
        y.append(state[1])
    return x, y

def plot_trajectory(batch, env, nb, save_figure=True):
    if env.observation_space.shape[0] < 2:
        raise(ValueError("Observation space of dimension {}, should be at least 2".format(env.observation_space.shape[0])))

    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for episode in batch.episodes:
        x, y = episode_to_traj(episode)
        plt.scatter(x, y, c=range(1, len(episode.state_pool) + 1), s=3)
    figname = 'trajectory_' + str(nb) + '.pdf'
    final_show(save_figure, False, figname, x_label, y_label, "Trajectory", '/plots/')