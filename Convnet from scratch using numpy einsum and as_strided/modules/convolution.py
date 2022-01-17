import numpy as np
from numpy.lib.stride_tricks import as_strided

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        x = np.pad(x, pad_width= ((0,0),(0,0),(self.padding, self.padding), (self.padding, self.padding)))
        N, C, H, W = x.shape

        S_N, S_C, S_H, S_W = x.strides

        Hout = (H-self.kernel_size)//self.stride + 1
        Wout = (W-self.kernel_size)//self.stride + 1

        x_view = as_strided(x, shape = (N, C, Hout, Wout, self.kernel_size, self.kernel_size), \
                       strides = (S_N, S_C, S_H*self.stride, S_W*self.stride, S_H, S_W))

        out = np.einsum('nchwkl,ockl->nohw', x_view, self.weight) # out dim: (N, Cout, Hout, Wout)

        out += self.bias[None, :, None, None]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape #padded
        S_N, S_C, S_H, S_W = x.strides
        Hout = (H-self.kernel_size)//self.stride + 1
        Wout = (W-self.kernel_size)//self.stride + 1

        x_view = as_strided(x, shape=(N, C, Hout, Wout, self.kernel_size, self.kernel_size), \
                            strides=(S_N, S_C, S_H * self.stride, S_W * self.stride, S_H, S_W))

        # dout shape: (N, cout, hout, wout)
        # x_view shape: (N, Cin, Hout, Wout, self.kernel_size, self.kernel_size)
        # kernel shape: (Cout, Cin, kernel_size, kernel_size)
        # print(Hout)
        # print(dout.shape[2])
        # print(Wout)
        # print(dout.shape[3])
        # dx_view = np.zeros(x_view.shape)
        dx = np.zeros(x.shape)
        # dout shape: (N, self.out_channels, H', W')
        for i in range(dout.shape[0]):
            for j in range(dout.shape[1]):  # for each Cout (kernel count)
                for k in range(dout.shape[2]): # H_out
                    for l in range(dout.shape[3]): #W_out
                        # print(l)
                        # print(self.weight[j].shape)
                        # print(dx[i,:, k*self.stride:k*self.stride+self.kernel_size, l*self.stride:l*self.stride + self.kernel_size].shape)
                        # print("-----")
                        dx[i,:, k*self.stride:k*self.stride+self.kernel_size, l*self.stride:l*self.stride + self.kernel_size] += dout[i,j,k,l]*self.weight[j]
                        # shape: (Cin, kernelsize, kernelsize)
                        # dx_view[i,:,k,l,:,:] += dout[i,j,k,l]*self.weight[j]

        self.dx = dx[:,:,1:-1,1:-1]

        self.dw = np.einsum('nohw,nchwkl->ockl', dout, x_view)
        # dout x input cross correlation

        # self.bias shape: (self.out_channels)
        doutdb = np.ones((x_view.shape[0], x_view.shape[2], x_view.shape[3])) # the axes upon each kernel's bias is shared
        self.db = np.einsum('nohw,nhw->o', dout, doutdb)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################