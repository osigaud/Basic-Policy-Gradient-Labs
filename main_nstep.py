from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import exploit_nstep
from main_pg import create_data_folders, set_files


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
        simu.env.set_file_name("nstep_" + str(n) + '_' + simu.env_name)
        policy_loss_file, critic_loss_file = set_files("nstep_" + str(n), simu.env_name)
        print("nstep : ", params.nstep)
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
            # plot_policy(policy, simu.env, True, simu.name, "nstep", '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, "nstep", '_ante_', j)

            simu.train(pw, params, policy, critic, policy_loss_file, critic_loss_file, "nstep")
            plot_policy(policy, simu.env, True, simu.env_name, "nstep", '_post_', j, plot=False)
            plot_critic(simu, critic, policy, "nstep", '_post_', j)
    chrono.stop()


def main():
    args = get_args()
    print(args)
    create_data_folders()
    args.gradients = ['sum', 'discount', 'normalize']
    study_nstep(args)
    exploit_nstep(args)


if __name__ == '__main__':
    main()
