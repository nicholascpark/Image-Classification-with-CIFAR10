import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        # input is an image of size: 32 x 32 x 3 channels
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (7,7), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2), stride=2) # 128 images with size 13 x 13 x 32
        self.fc1 = nn.Linear(5408, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # print("before conv:", x.shape)
        x = self.conv1(x)
        # print("before relu:", x.shape)
        x = self.relu1(x)
        # print("before pool:", x.shape)
        x = self.pool1(x)
        # print("before flatten:", x.shape)
        x = torch.flatten(x, 1)
        # print("before linear:", x.shape)
        x = self.fc1(x)
        # print("after softmax:", x.shape)
        # outs = nn.functional.log_softmax(x, dim=1) --- already included in criterion
        outs = x
        # print("after softmax:", x.shape)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs