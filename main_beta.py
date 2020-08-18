from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from exploit_results import main_exploit
from main_pg import create_data_folders, set_files


def study_beta(params):
    simu = make_simu_from_params(params)
    for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
        policy_loss_file, critic_loss_file = set_files(str(beta), simu.name)
        simu.env.set_file_name(str(beta) + '_' + simu.name)
        for i in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            pw = PolicyWrapper(policy, params.team_name, simu.name)
            simu.train(pw, params, simu.env, policy, critic, policy_loss_file, critic_loss_file, "beta")


def main():
    args = get_args()
    print(args)
    create_data_folders()
    args.gradients = ['sum', 'discount', 'normalize']
    study_beta(args)
    main_exploit(args)


if __name__ == '__main__':
    main()
