import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    numclass = len(cls_num_list)
    per_cls_weights = (1 - beta)/(1 - np.power(beta, cls_num_list)) # (1-b)/(1-b^nc) = alpha = 1/en
    per_cls_weights = per_cls_weights * numclass / sum(per_cls_weights)
    print(per_cls_weights/numclass)
    per_cls_weights = torch.from_numpy(per_cls_weights)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        m = torch.nn.LogSoftmax()
        logp = m(input)
        logpt = logp[range(input.shape[0]), target]
        pt = torch.exp(logpt)
        fl = -((1-pt) ** self.gamma)*logpt
        # print(self.weight)
        alpha = self.weight[target].float()
        # print(alpha)
        loss = torch.dot(fl,alpha) / input.shape[0]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
