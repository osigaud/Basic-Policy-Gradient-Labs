import random
from itertools import count
from batch import Episode, Batch

"""
These functions are used to generate expert trajectories on Mountain Car environments
and to learn a policy by regression from these trajectories, so as to circumvent a deceptive gradient effect
that makes standard policy gradient very inefficient on these environments.
"""


def perform_expert_episodes(simu, batch, render=False):
    for e in range(20):
        state = simu.reset(render)

        episode = Episode()
        for t in range(50):
            coin = random.random() / 20
            action = [-1.0 + coin]
            state, reward, done = simu.take_step(state, action, episode, render)

        for _ in count():
            coin = random.random() / 10
            action = [1.0 - coin]
            state, reward, done = simu.take_step(state, action, episode, render)

            if done:
                batch.add_episode(episode)
                break
    return batch


def regress(simu, policy, render=False) -> None:
    batch = Batch()
    batch = perform_expert_episodes(simu, batch, render)
    # print("size: ", batch.size())
    batch.train_regress_actor(policy)
