import numpy as np

class Linear:
    '''
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    '''
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None


    def forward(self, x):
        '''
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        '''
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        x_flat = np.reshape(x, (x.shape[0],-1))
        out = np.matmul(x_flat, self.weight) + self.bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        # print(out.dtype)
        return out

    def backward(self, dout):
        '''
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        # print("linear backward in:", dout.shape)
        x_flat = np.reshape(x, (x.shape[0],-1))

        dLdout = dout
        doutdx = self.weight.T
        dLdx = np.matmul(dLdout, doutdx)
        self.dx = dLdx.reshape(x.shape)

        doutdw = x_flat.T
        dLdw = np.matmul(doutdw, dLdout)
        self.dw = dLdw

        doutdb = np.ones(x.shape[0])
        dLdb = np.matmul(doutdb, dLdout)
        self.db = dLdb
        # print("linear backward out:", self.dx.shape)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


