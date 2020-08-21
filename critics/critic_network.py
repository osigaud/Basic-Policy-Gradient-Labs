import numpy as np
from policies import GenericNet
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler


def make_subsets_samplers(dataset):
    """
    Splits a dataset into a validation part (20%) and a training part (80%)
    :param dataset: the initial dataset
    :return: a random sampler for each of the sub-datasets (training and validation)
    """
    validation_split = .2
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)


class CriticNetwork(GenericNet):
    """
    A generic class for all kinds of network critics
    TODO: methods for training from a pytorch dataset would deserve to be refactored
    """
    def __init__(self):
        super(CriticNetwork, self).__init__()

    def update(self, loss) -> None:
        """
        Updates the network given a loss value
        :param loss: the loss to be applied
        :return: nothing
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_from_loader(self, train_loader):
        """
        Train a critic from a python dataset structure
        :param train_loader: the loader built from a dataset
        :return: the obtained critic loss
        """
        for step, (batch_s, batch_a, batch_t) in enumerate(train_loader):  # for each training step
            state = batch_s.data.numpy()
            action = batch_a.data.numpy()
            target = batch_t
            critic_loss = self.compute_loss_to_target(state, action, target)
            self.update(critic_loss)
            return critic_loss

    def compute_validation_loss(self, validation_loader, train=False):
        """
        Compute the validation loss from samples of a pytorch dataset that have been put aside of training
        The computation is performed a number of times
        :param validation_loader: the validation loader built from a dataset
        :param train: whether we train the critic while computing the validation loss (should be False)
        :return: the obtained vector of losses
        """
        losses = []
        for step, (batch_s, batch_a, batch_t) in enumerate(validation_loader):  # for each training step
            state = batch_s.data.numpy()
            action = batch_a.data.numpy()
            target = batch_t
            critic_loss = self.compute_loss_to_target(state, action, target)
            if train:
                self.update(critic_loss)
            critic_loss = critic_loss.data.numpy()
            losses.append(critic_loss)
        return np.array(losses)

    def compute_valid_mc(self, params, dataset, critic_loss_file, trace_loss=False, save_best=True):
        """
        Compute the validation loss from samples of a pytorch dataset that have been put aside of training
        Using a Monte Carlo method
        The computation is performed a number of times
        :param params: hyper-parameters of the experiments. Here, specifying the use of the dataset
        :param dataset: the dataset from which to train the critic
        :param critic_loss_file: the file where to put the obtained loss
        :param trace_loss: whether we want to record the loss
        :param save_best: whether we save the critic whose validation loss is the lowest
        :return:
        """
        train_sampler, valid_sampler = make_subsets_samplers(dataset)

        best_loss = 1000.0
        t_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=train_sampler)
        v_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=valid_sampler)
        for epoch in range(params.nb_batches):
            self.train_loss(t_loader)
            losses = self.compute_validation_loss(v_loader)
            critic_loss = losses.mean()
            if trace_loss:
                critic_loss_file.write(str(epoch) + " " + str(critic_loss) + "\n")
            if save_best and best_loss > critic_loss:
                best_loss = critic_loss
                self.save_model('./critics/' + params.env_name + '#' + params.team_name + '#' + str(critic_loss) + '.pt')
        return critic_loss  # returns the last critic loss over the nb_batches

    def update_mc(self, params, dataset, train, save_best=True):
        """
        Update the critic from a dataset using a Monte Carlo method
        :param params: hyper-parameters of the experiments. Here, specifying the use of the dataset
        :param dataset: the dataset from which to train the critic
        :param train: whether we should train while computing the validation loss (should be False)
        :param save_best: whether we save the critic whose validation loss is the lowest
        :return: the last critic loss over the nb_batches (it would be better to return a mean)
        """
        best_loss = 10000
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, shuffle=params.shuffle, num_workers=params.nb_workers, )
        for epoch in range(params.nb_batches):
            losses = self.compute_validation_loss(loader, train)
            critic_loss = losses.mean()
            if save_best and best_loss > critic_loss:
                best_loss = critic_loss
                # print("cpt: ", epoch, " loss : ", loss)
                self.save_model('./critics/' + params.env_name + '#' + params.team_name + '#' + str(critic_loss) + '.pt')
        return critic_loss  # returns the last critic loss over the nb_batches

    def update_td(self, params, dataset, train):
        """
        Update the critic from a dataset using a temporal difference method
        :param params: hyper-parameters of the experiments. Here, specifying the use of the dataset
        :param dataset: the dataset from which to train the critic
        :param train: whether we should train while computing the validation loss (should be False)
        :return: the mean over the obtained loss
        """
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, shuffle=params.shuffle, num_workers=params.nb_workers, )
        losses = self.compute_validation_loss(loader, train)
        loss = losses.mean()
        return loss

    def compute_valid_td(self, params, dataset):
        """
        Compute the validation loss from samples of a pytorch dataset that have been put aside of training
        Using a Monte Carlo method
        The computation is performed a number of times
        :param params: hyper-parameters of the experiments. Here, specifying the use of the dataset
        :param dataset: the dataset from which to train the critic
        :return: the mean over the obtained losses
        """
        train_sampler, valid_sampler = make_subsets_samplers(dataset)
        t_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=train_sampler)
        v_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=valid_sampler)
        self.train_from_loader(t_loader)
        losses = self.compute_validation_loss(v_loader)
        loss = losses.mean()
        return loss