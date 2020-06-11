# started from Finspire13 /pytorch-policy-gradient-example

import os
from chrono import Chrono
from policies import BernoulliPolicy, NormalPolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from simu import make_simu_from_params
from visu.visu_policies import plot_policy
from exploit_results import main_exploit


def set_files(study_name, env_name):
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    value_loss_name = "data/save/value_loss_" + study_name + '_' + env_name + ".txt"
    value_loss_file = open(value_loss_name, "w")
    return policy_loss_file, value_loss_file


def study_beta(params):
    simu = make_simu_from_params(params)
    for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
        policy_loss_file, value_loss_file = set_files(str(beta), simu.name)
        simu.env.set_file_name(str(beta) + '_' + simu.name)
        for i in range(params.nb_repet):
            policy = BernoulliPolicy(1, 24, 36, 1, params.lr_actor)
            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            pw = PolicyWrapper(policy, params.team_name, simu.name)
            simu.perform_episodes(pw, params, simu.env, policy, critic, policy_loss_file, value_loss_file, "beta")


def study_pg(params):
    chrono = Chrono()
    study = params.gradients  # ["sum", "discount", "normalize", "baseline"]  #
    simu = make_simu_from_params(params)
    for i in range(len(study)):
        simu.env.set_file_name(study[i] + '_' + simu.name)
        policy_loss_file, value_loss_file = set_files(study[i], simu.name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            else:
                print("main PG: unknown policy type: ", params.policy_type)
            pw = PolicyWrapper(policy, params.team_name, simu.name)
            # plot_policy(policy, simu.env, simu.name, study[i], '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, study[i], '_ante_', j)

            simu.perform_episodes(pw, params, policy, critic, policy_loss_file, value_loss_file, study[i])
        plot_policy(policy, simu.env, simu.name, study[i], '_post_', j, plot=False)
        plot_critic(simu, critic, policy, study[i], '_post_', j)
        critic.save_model('data/critics/' + params.env_name + '#' + params.team_name + '#' + study[i] + str(j) + '.pt')
    chrono.stop()
    return study


def study_regress(params) -> None:
    simu = make_simu_from_params(params)
    simu.env.set_file_name('regress_' + simu.name)
    policy_loss_file, value_loss_file = set_files('regress', simu.name)
    for j in range(params.nb_repet):
        policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
        pw = PolicyWrapper(policy, "team", simu.name)
        act_size = simu.env.action_space.shape[0]
        critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
        plot_policy(policy, simu.env, simu.name, 'regress', '_ante_', j, plot=False)
        plot_critic(simu, critic, policy, 'regress', '_ante_', j)

        simu.regress(simu.env, policy, simu.name, params.render)
        simu.perform_episodes(pw, params, simu.env, policy, critic, policy_loss_file, value_loss_file, "regress")
        plot_policy(policy, simu.env, simu.name, 'regress', '_post_', j, plot=False)
        plot_critic(simu, critic, policy, 'regress', '_post_', j)


def study_nstep(params):
    """
    Trying to learn the policy using nstep return
    Not to be confused with learning the critic with nstep return
    :param params: the experimental setup parameters
    :return: nothing (creates output files)
    """
    chrono = Chrono()
    simu = make_simu_from_params(params)
    for n in [1, 5, 10, 15, 20]:
        params.nstep = n
        simu.env.set_file_name("nstep_" + str(n) + '_' + simu.name)
        policy_loss_file, value_loss_file = set_files("nstep_" + str(n), simu.name)
        print("nstep : ", params.nstep)
        for j in range(params.nb_repet):
            policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            pw = PolicyWrapper(policy, params.team_name, simu.name)
            # plot_policy(policy, simu.env, simu.name, "nstep", '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, "nstep", '_ante_', j)

            simu.perform_episodes(pw, params, policy, critic, policy_loss_file, value_loss_file, "nstep")
            plot_policy(policy, simu.env, simu.name, "nstep", '_post_', j, plot=False)
            plot_critic(simu, critic, policy, "nstep", '_post_', j)
    chrono.stop()


def main():
    args = get_args()
    print(args)

    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")

    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")

    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')

    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')

    args.gradients = ['sum', 'discount', 'normalize']
    if args.study_name == "pg":
        study_pg(args)
        main_exploit(args)
    elif args.study_name == "regress":
        study_regress(args)
        main_exploit(args)
    elif args.study_name == "beta":
        study_beta(args)
        main_exploit(args)
    elif args.study_name == "nstep":
        study_nstep(args)
        main_exploit(args)
    else:
        print("cas d'étude inconnu")


if __name__ == '__main__':
    main()

"""
"""
