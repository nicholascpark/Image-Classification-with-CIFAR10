import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden_size, self. num_classes)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # print("before flatten: ", x.shape)
        x = torch.flatten(x, 1)
        # print("after flatten: ", x.shape)
        x = self.fc1(x)
        # print("after fc1:", x.shape)
        x = self.sigmoid(x)
        # print("after sigmoid:", x.shape)
        x = self.fc2(x)
        # print("after fc2:", x.shape)
        out = x

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out