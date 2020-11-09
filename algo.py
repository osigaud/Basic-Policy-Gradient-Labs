

class Algo:
    """
    The Algo class is an intermediate structure to unify various algorithms by collecting hyper-parameters
    """
    def __init__(self, study_name, critic_estim_method, policy, critic, gamma, beta, n):
        self.study_name = study_name
        self.critic_estim_method = critic_estim_method
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.beta = beta
        self.n = n

    def prepare_batch(self, batch) -> None:
        """
        Applies reward transformations into the batch to prepare the computation of some gradient over these rewards
        :param batch: the batch on which we train
        :return: nothing
        """
        assert self.study_name in ['beta', 'sum', 'discount', 'normalize', 'baseline', 'nstep'], 'unsupported study name'
        if self.study_name == "beta":
            batch.exponentiate_rewards(self.beta)
        elif self.study_name == "sum":
            batch.sum_rewards()
        elif self.study_name == "discount":
            batch.discounted_sum_rewards(self.gamma)
        elif self.study_name == "normalize":
            batch.normalize_rewards(self.gamma)
        elif self.study_name == "baseline":
            batch.discounted_sum_rewards(self.gamma)
            batch.substract_baseline(self.critic)
        elif self.study_name[:5] == "nstep":
            batch.nstep_return(self.n, self.gamma, self.critic)


    def train_critic_from_dataset(self, batch, params):
        """
        Train the critic from a dataset
        :param batch: the batch on which we train it (is transformed into a pytorch dataset
        :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
        :return: the critic training loss
        """
        assert self.critic_estim_method in ['td', 'mc', 'nstep'], 'unsupported critic estimation method'
        if self.critic_estim_method == "td":
            dataset = batch.prepare_dataset_td(params, self.policy, self.critic)
            return self.critic.update_td(params, dataset, True)
        elif self.critic_estim_method == "mc":
            dataset = batch.prepare_dataset_mc(self.gamma)
            return self.critic.update_mc(params, dataset, True, save_best=False)
        elif self.critic_estim_method == "nstep":
            dataset = batch.prepare_dataset_td(params, self.policy, self.critic, "nstep")
            return self.critic.compute_valid_td(params, dataset)


    def train_critic_from_batch(self, batch):
        """
        Train the critic from a batch
        :param batch: the batch on which we train it
        :return: the critic training loss
        """
        if self.critic_estim_method == "td":
            return batch.train_critic_td(self.gamma, self.policy, self.critic, True)
        elif self.critic_estim_method == "mc":
            return batch.train_critic_mc(self.gamma, self.critic, 0, True)
        elif self.critic_estim_method == "nstep":
            return batch.train_critic_mc(self.gamma, self.critic, self.n, True)
        else:
            print("Algo train_policy_batch : unknown critic estim method : ", self.critic_estim_method)
        return 0
