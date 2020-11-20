import numpy as np
import gym
import torch
import os


from chrono import Chrono
from critics.q_net import QNet, calc_target
from policies.policy_net import PolicyNet
from memory import ReplayBuffer
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import plot_results
from wrappers.policy_wrapper import PolicyWrapper

lr_pi = 0.0005
lr_q = 0.001
batch_size = 32


def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")
    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")
    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')
    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')


def set_files(study_name, env_name):
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + \
        study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + \
        study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def main(params) -> None:
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    
    obs_size = env.observation_space.shape[0]

    chrono = Chrono()

    for j in range(params.nb_repet):
        env.reinit()

        # Initialise the policy/actor
        memory = ReplayBuffer()
        policy = PolicyNet(lr_pi)

        score = 0.0
        print_interval = 20

        pw = PolicyWrapper(policy, params.policy_type, env_name,
                           params.team_name, params.max_episode_steps)
        plot_policy(policy, env, True, env_name, 'SAC', '_ante_', j, plot=False)

        # Initialise the critics
        q1, q2, q1_target, q2_target = QNet(
            lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        for n_epi in range(300):
            s = env.reset()
            done = False

            # equivalent d'une traj aka 1 épisode
            while not done:
                a, log_prob = policy(torch.from_numpy(
                    s).float())    # action selection
                s_prime, r, done, info = env.step([2.0*a.item()])
                # add of the global state in the replay buffer
                memory.put((s, a.item(), r/10.0, s_prime, done))
                score += r
                s = s_prime

            if memory.size() > 1000:
                for i in range(20):
                    # construction d'une fraction de traj
                    mini_batch = memory.sample(batch_size)
                    # calcule la cible pour la fonction Q
                    td_target = calc_target(
                        policy, q1_target, q2_target, mini_batch)
                    # entraine la 1ere critique
                    q1.train_net(td_target, mini_batch)
                    # entraine la 2e critique
                    q2.train_net(td_target, mini_batch)
                    # entraine la politque (= acteur)
                    entropy = policy.train_net(q1, q2, mini_batch)
                    q1.soft_update(q1_target)   # update of the 1st target
                    q2.soft_update(q2_target)   # update of the 2nd target

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(
                    n_epi, score/print_interval, policy.log_alpha.exp()))
                score = 0.0
        
        plot_policy(policy, env, True, env_name, 'SAC', '_post_', j, plot=False)
        plot_critic(simu, q1, policy, 'SAC', '_post_', j)
        q1.save_model(
            'data/critics/' + params.env_name + '#' + params.team_name + '#' + study[i] + str(j) + '.pt')

    env.close()


if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    main(args)
    plot_results(args)
