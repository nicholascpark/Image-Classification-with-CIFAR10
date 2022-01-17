import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), stride=1, padding=1)
        #  32 x 32 x 16
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), stride=1, padding=1)
        #  32 x 32 x 16
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features = 16)

        self.pool1 = nn.MaxPool2d((2,2), stride=1)
        # 31 x 31 x 16
        self.conv3= nn.Conv2d(in_channels= 16, out_channels=32, kernel_size= (3,3), stride = 1, padding= 1)
        # 31 x 31 x 32
        self.conv4= nn.Conv2d(in_channels= 32, out_channels=32, kernel_size= (3,3), stride = 1, padding= 1)
        # 31 x 31 x 32
        self.relu2 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(num_features = 32)

        self.pool2 = nn.MaxPool2d((2,2), stride=2)
        # 15 x 15 x 32
        self.conv5 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= (3,3), stride = 1, padding= 1)
        # 15 x 15 x64
        self.conv6 = nn.Conv2d(in_channels= 64, out_channels=64, kernel_size= (3,3), stride = 1, padding= 1)
        # 15 x 15 x64
        self.conv7 = nn.Conv2d(in_channels= 64, out_channels=64, kernel_size= (3,3), stride = 1, padding= 1)
        # 15 x 15 x 64


        self.relu3 = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(num_features = 64)

        self.pool3 = nn.MaxPool2d((2,2), stride=1)
        # 14 x 14 x 64
        self.conv8 = nn.Conv2d(in_channels= 64, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        # 14 x 14 x 64
        self.conv9 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        # 14 x 14 x 64
        self.conv10 = nn.Conv2d(in_channels= 128,  out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        # 14 x 14 x 64
        self.relu4 = nn.ReLU()

        self.bn4 = nn.BatchNorm2d(num_features = 128)
        self.pool4 = nn.MaxPool2d((2,2), stride=2)
        # 7 x 7 x 128

        self.conv11 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        self.conv12 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        self.conv13 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        # 7 x 7 x 128

        self.relu5 = nn.ReLU()

        self.bn5 = nn.BatchNorm2d(num_features = 128)
        self.pool5 = nn.MaxPool2d((2,2), stride=1)
        # 6 x 6 x 128

        self.conv14 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        self.conv15 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        self.conv16 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= (3,3), stride = 1, padding= 1)
        # 6 x 6 x 128

        self.relu6 = nn.ReLU()

        self.bn6 = nn.BatchNorm2d(num_features = 128)
        self.pool6 = nn.MaxPool2d((2,2), stride=2)
        # 3 x 3 x 128


        self.fc1 = nn.Linear(1152, 1152)
        # sigmoid?
        self.fc2 = nn.Linear(1152, 1152)
        self.fc3 = nn.Linear(1152, 10)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu1(x)

        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.relu3(x)

        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.relu5(x)
        x = self.bn5(x)
        x = self.pool5(x)

        x = self.conv14(x) #
        x = self.conv15(x) #
        x = self.conv16(x) #

        x = self.relu6(x)
        x = self.bn6(x)
        x = self.pool6(x)


        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.dropout3(x)
        outs = self.fc3(x)
        # print(x.shape)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs