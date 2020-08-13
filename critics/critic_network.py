import numpy as np
from policies import GenericNet
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler


def make_subsets_samplers(dataset):
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
    def __init__(self):
        super(CriticNetwork, self).__init__()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_loss(self, train_loader):
        for step, (batch_s, batch_a, batch_t) in enumerate(train_loader):  # for each training step
            state = batch_s.data.numpy()
            action = batch_a.data.numpy()
            target = batch_t
            self.compute_target_loss(state, action, target, True)

    def compute_validation_loss(self, validation_loader, train=False):
        losses = []
        for step, (batch_s, batch_a, batch_t) in enumerate(validation_loader):  # for each training step
            state = batch_s.data.numpy()
            action = batch_a.data.numpy()
            target = batch_t
            value_loss = self.compute_target_loss(state, action, target, train)
            loss = value_loss.data.numpy()
            losses.append(loss)
        return np.array(losses)

    def update_valid_mc(self, params, dataset, value_loss_file, trace_loss=False, save_best=True):
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
            loss = losses.mean()
            if trace_loss:
                value_loss_file.write(str(epoch) + " " + str(loss) + "\n")
            if save_best and best_loss > loss:
                best_loss = loss
                # print("cpt: ", epoch, " loss : ", best_loss)
                self.save_model('./critics/' + params.env_name + '#' + params.team_name + '#' + str(loss) + '.pt')
        return loss  # renvoie la dernière loss sur les nb_batches
        # return range(len(all_losses)), all_losses

    def update_mc(self, params, dataset, train, save_best=True):
        best_loss = 10000
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, shuffle=params.shuffle, num_workers=params.nb_workers, )
        for epoch in range(params.nb_batches):
            losses = self.compute_validation_loss(loader, train)
            loss = losses.mean()
            if save_best and best_loss > loss:
                best_loss = loss
                # print("cpt: ", epoch, " loss : ", loss)
                self.save_model('./critics/' + params.env_name + '#' + params.team_name + '#' + str(loss) + '.pt')
        # return range(params.nb_batches), losses
        return loss  # renvoie la dernière loss sur les nb_batches

    def update_td(self, params, dataset, train):
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, shuffle=params.shuffle, num_workers=params.nb_workers, )
        losses = self.compute_validation_loss(loader, train)
        loss = losses.mean()
        return loss

    def update_valid_td(self, params, dataset):
        train_sampler, valid_sampler = make_subsets_samplers(dataset)
        t_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=train_sampler)
        v_loader = data.DataLoader(
            dataset=dataset,
            batch_size=params.batch_size, num_workers=params.nb_workers, sampler=valid_sampler)
        train_loss(t_loader, critic)
        losses = self.compute_validation_loss(v_loader)
        loss = losses.mean()
        return loss