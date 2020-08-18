from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from exploit_results import main_exploit
from main_pg import create_data_folders, set_files
from mountain_car_expert import regress


def study_regress(params) -> None:
    chrono = Chrono()
    simu = make_simu_from_params(params)
    simu.env.set_file_name('regress_' + simu.name)
    policy_loss_file, critic_loss_file = set_files('regress', simu.name)
    for j in range(params.nb_repet):
        simu.env.reinit()
        if params.policy_type == "bernoulli":
            policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
        elif params.policy_type == "normal":
            policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
        pw = PolicyWrapper(policy, "team", simu.name)
        act_size = simu.env.action_space.shape[0]
        critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
        plot_policy(policy, simu.env, True, simu.name, 'regress', '_ante_', j, plot=False)
        # plot_critic(simu, critic, policy, 'regress', '_ante_', j)

        regress(simu, policy, params.policy_type, params.render)
        simu.train(pw, params, policy, critic, policy_loss_file, critic_loss_file, "regress")
        plot_policy(policy, simu.env, True, simu.name, 'regress', '_post_', j, plot=False)
        # plot_critic(simu, critic, policy, 'regress', '_post_', j)
    chrono.stop()

def main():
    args = get_args()
    args.study_name = "regress"
    print(args)
    create_data_folders()
    study_regress(args)
    main_exploit(args)

if __name__ == '__main__':
    main()
