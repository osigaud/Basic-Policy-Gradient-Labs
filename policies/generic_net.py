import torch
import torch.nn as nn
import torch.nn.functional as func


class GenericNet(nn.Module):
    def __init__(self):
        super(GenericNet, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

    def save_model(self, filename):
        torch.save(dict(model=self, model_state=self.state_dict(), optimizer=self.optimizer.state_dict()), filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        net = checkpoint["model"]
        net.load_state_dict(checkpoint['model_state'])
        net.optimizer.load_state_dict(checkpoint['optimizer'])
        net.eval()
        return net

    def train_regress(self, state, action):
        proposed_action = self.forward(state)
        loss = func.mse_loss(proposed_action, action)
        self.update(loss)
        return loss




# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

