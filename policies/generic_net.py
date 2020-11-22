import torch
import torch.nn as nn


class GenericNet(nn.Module):
    """The super class of all policy and critic networks.

    Contains general behaviors like loading and saving, and updating from a loss

    Attributes:
        loss_func: The standard loss function used is the Mean Squared Error (MSE)
    """
    def __init__(self):
        super(GenericNet, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def save_model(self, filename) -> None:
        """Save a neural network model into a file.

        Args:
            filename (str): The filename, including the path.
        """
        torch.save(self, filename)

    def load_model(self, filename):
        """Load a neural network model from a file.

        Args:
            filename (str): The filename, including the path.

        Returns:
            The resulting pytorch network
        """
        net = torch.load(filename)
        net.eval()
        return net

    def update(self, loss) -> None:
        """Apply a loss to a network using gradient backpropagation.

        Args:
            loss (torch.Tensor): The applied loss.
        """
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

