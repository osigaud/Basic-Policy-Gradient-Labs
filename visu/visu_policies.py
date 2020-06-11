import os
import numpy as np
import matplotlib.pyplot as plt
import random


def final_show(save_figure, plot, figname, x_label, y_label, title, dir):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_figure:
        directory = os.getcwd() + '/data' + dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figname)
    if plot:
        plt.show()
    plt.close()


def plot_policy(policy, env, name, study_name, default_string, num, plot=False):
    obs_size = env.observation_space.shape[0]
    actor_picture_name = str(num) + '_actor_' + study_name + '_' + default_string +  name + '.pdf'
    if obs_size == 1:
        plot_stoch_policy_1D(policy, env, plot, figname=actor_picture_name)
    elif obs_size == 2:
        plot_stoch_policy_2D(policy, env, plot, figname=actor_picture_name)
    else:
        plot_stoch_policy_ND(policy, env, plot, figname=actor_picture_name)


# visualization of the policy for a 1D environment like 1D Toy with continuous actions
def plot_stoch_policy_1D(policy, env, plot=True, figname="policy_1D.pdf", save_figure=True, definition=50):
    if env.observation_space.shape[0] != 1:
        raise(ValueError("The observation space dimension is {}, should be 1".format(env.observation_space.shape[0])))

    x_min = env.observation_space.low[0]
    x_max = env.observation_space.high[0]

    states = []
    actions = []
    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        state = np.array([x])
        action = policy.select_action(state)
        states.append(state)
        actions.append(action)

    plt.figure(figsize=(10, 10))
    plt.plot(states, actions)
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "1D Policy", '/plots/')


# visualization of the policy for a 2D environment like continuous mountain car.
def plot_stoch_policy_2D(policy, env, plot=True, figname='stoch_actor.pdf', save_figure=True, definition=50):
    """Portrait the actor"""
    if env.observation_space.shape[0] != 2:
        raise(ValueError("Observation space dimension {}, should be 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            state = np.array([[x, y]])
            portrait[definition - (1 + index_y), index_x] = policy.select_action(state)
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", '/plots/')


# visualization of the policy for a 2D environment like continuous mountain car.
def plot_proba_policy(policy, env, plot=True, figname='proba_actor.pdf', save_figure=True, definition=50):
    """Portrait the actor"""
    if env.observation_space.shape[0] != 2:
        raise(ValueError("Observation space dimension {}, should be 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            state = np.array([[x, y]])
            probs = policy.forward(state)
            action = probs.data.numpy().astype(float)
            # print(probs, action)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", '/plots/')


# visualization of the policy for a ND environment like pendulum
def plot_stoch_policy_ND(policy, env, plot=True, figname='stoch_actor.pdf', save_figure=True, definition=50):
    """Portrait the actor"""
    if env.observation_space.shape[0] <= 2:
        raise(ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"

    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[1], state_max[1], num=definition)):
            state = np.array([[x, y]])
            for i in range(2, len(state_min)):
                z = random.random() - 0.5
                state = np.append(state, z)
            action = policy.select_action(state)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    # print(state_min[0], ";", state_max[0], ";", state_min[1], ";", state_max[1], state_min[2], ";", state_max[2], state_min[3], ";", state_max[3])
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[1], state_max[1]], aspect='auto')
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", '/plots/')
