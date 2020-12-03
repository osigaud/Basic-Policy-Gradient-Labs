# started from Finspire13 /pytorch-policy-gradient-example

import os
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import plot_results
import gym


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
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def study_pg(params) -> None:
    """
    Start a study of the policy gradient algorithms
    :param params: the parameters of the study
    :return: nothing
    """
    assert params.policy_type in ['bernoulli', 'normal', 'squashedGaussian', 'discrete'], 'unsupported policy type'
    chrono = Chrono()
    # cuda = torch.device('cuda')
    study = params.gradients
    simu = make_simu_from_params(params)
    for i in range(len(study)):
        simu.env.set_file_name(study[i] + '_' + simu.env_name)
        policy_loss_file, critic_loss_file = set_files(study[i], simu.env_name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "discrete":
                if isinstance(simu.env.action_space , gym.spaces.box.Box):
                    nb_actions = int(simu.env.action_space.high[0] - simu.env.action_space.low[0] +1)
                    print("Error : environment action space is not discrete :"+str(simu.env.action_space))
                else :
                    nb_actions = simu.env.action_space.n
                policy = DiscretePolicy(simu.obs_size, 24, 36, nb_actions, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "squashedGaussian":
                policy = SquashedGaussianPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            # policy = policy.cuda()
            pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
            plot_policy(policy, simu.env, True, simu.env_name, study[i], '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, study[i], '_ante_', j)

            simu.train(pw, params, policy, critic, policy_loss_file, critic_loss_file, study[i])
            plot_policy(policy, simu.env, True, simu.env_name, study[i], '_post_', j, plot=False)
            if False:
                if params.policy_type == "normal":
                    plot_normal_histograms(policy, j, simu.env_name)
                else:
                    plot_weight_histograms(policy, j, simu.env_name)
        plot_critic(simu, critic, policy, study[i], '_post_', j)
        critic.save_model('data/critics/' + params.env_name + '#' + params.team_name + '#' + study[i] + str(j) + '.pt')
    chrono.stop()


if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    study_pg(args)
    plot_results(args)
