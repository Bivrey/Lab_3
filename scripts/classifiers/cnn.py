from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 8, 8), num_filters=32, filter_size=7,
             hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        F = num_filters
        HH = filter_size
        WW = filter_size
        H_hid = hidden_dim

        self.params['W1'] = np.random.randn(F, C, HH, WW) * weight_scale
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = np.random.randn(F * (H // 2) * (W // 2), H_hid) * weight_scale
        self.params['b2'] = np.zeros(H_hid)
        self.params['W3'] = np.random.randn(H_hid, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        N = X.shape[0]  
        X = X.reshape(N, 1, 8, 8)
        a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        a2, cache2 = affine_relu_forward(a1, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        da2, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
        da1, grads['W2'], grads['b2'] = affine_relu_backward(da2, cache2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(da1, cache1)

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3

        return loss, grads
