import torch
import torch.nn as nn


class GenericNet(nn.Module):
    """
    The super class of all policy and critic networks
    Contains general behaviors like loading and saving, and updating from a loss
    The stardnard loss function used is the Mean Squared Error (MSE)
    """
    def __init__(self):
        super(GenericNet, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def save_model(self, filename) -> None:
        """
        Save a neural network model into a file
        :param filename: the filename, including the path
        :return: nothing
        """
        torch.save(self, filename)

    def load_model(self, filename):
        """
        Load a neural network model from a file
        :param filename: the filename, including the path
        :return: the resulting pytorch network
        """
        net = torch.load(filename)
        net.eval()
        return net

    def update(self, loss) -> None:
        """
        Apply a loss to a network using gradient backpropagation
        :param loss: the applied loss
        :return: nothing
        """
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

