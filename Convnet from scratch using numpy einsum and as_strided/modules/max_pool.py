import numpy as np
from numpy.lib.stride_tricks import as_strided

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        x = np.asarray(x)
        N, C, H, W = x.shape
        S_N, S_C, S_H, S_W = x.strides
        H_out = (H-self.kernel_size)//self.stride + 1
        W_out = (W-self.kernel_size)//self.stride + 1

        x_view = as_strided(x, shape = (N, C, H_out, W_out, self.kernel_size, self.kernel_size), \
                       strides = (S_N, S_C, S_H*self.stride, S_W*self.stride, S_H, S_W))
        out = np.max(x_view, axis=(4, 5)) # output dimension is x.shape[:4]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        # print(dout.shape)
        self.dx = np.zeros(x.shape)
        N, C, H, W = x.shape
        S_N, S_C, S_H, S_W = x.strides
        x_view = as_strided(x, shape = (N, C, H_out, W_out, self.kernel_size, self.kernel_size), \
                       strides = (S_N, S_C, S_H*self.stride, S_W*self.stride, S_H, S_W))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(H_out):
                    for l in range(W_out):
                        window_argmax = np.unravel_index(np.argmax(x_view[i,j,k,l]), x_view[i,j,k,l].shape) #tuple
                        self.dx[i,j][tuple(np.add([k*self.stride, l*self.stride], list(window_argmax)))] = dout[i,j,k,l]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################