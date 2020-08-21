import random
from itertools import count
from batch import Episode, Batch

"""
These functions are used to generate expert trajectories on Mountain Car environments
and to learn a policy by regression from these trajectories, so as to circumvent a deceptive gradient effect
that makes standard policy gradient very inefficient on these environments.
"""


def perform_expert_episodes_bangbang(simu, batch, render=False):
    """
    Build a batch of 20 expert episodes playing a simple bangbang policy (go left first, then go right)
    :param simu: the simulation
    :param batch: the batch to be filled
    :param render: whether the step is displayed or not (True or False)
    :return: the batch
    """
    for e in range(20):
        state = simu.reset(render)

        episode = Episode()
        for _ in range(50):
            action = [0.0]
            state, reward, done = simu.take_step(state, action, episode, render)

        for _ in count():
            action = [1.0]
            state, reward, done = simu.take_step(state, action, episode, render)

            if done:
                batch.add_episode(episode)
                break
    return batch


def perform_expert_episodes_continuous(simu, batch, render=False):
    """
    Build a batch of 20 expert episodes playing a simple policy close to bangbang (go left first, then go right)
    :param simu: the simulation
    :param batch: the batch to be filled
    :param render: whether the step is displayed or not (True or False)
    :return: the batch
    """
    for e in range(20):
        state = simu.reset(render)

        episode = Episode()
        for _ in range(50):
            variation = random.random() / 20
            action = [-1.0 + variation]
            state, reward, done = simu.take_step(state, action, episode, render)

        for _ in count():
            variation = random.random() / 10
            action = [1.0 - variation]
            state, reward, done = simu.take_step(state, action, episode, render)

            if done:
                batch.add_episode(episode)
                break
    return batch


def regress(simu, policy, policy_type, render=False) -> None:
    batch = Batch()
    simu.env.set_reward_flag(False)
    simu.env.set_duration_flag(False)
    if policy_type == "bernoulli":
        batch = perform_expert_episodes_bangbang(simu, batch, render)
    else:
        batch = perform_expert_episodes_continuous(simu, batch, render)
    # print("size: ", batch.size())
    batch.train_policy_through_regress(policy)
