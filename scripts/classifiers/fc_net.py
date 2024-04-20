from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.reg = reg
        self.params = {}
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
    
    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Forward pass
        hidden_layer, cache_hidden = affine_relu_forward(X, W1, b1)
        scores, cache_scores = affine_forward(hidden_layer, W2, b2)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))  # L2 regularization

        # Backward pass
        dhidden, dW2, db2 = affine_backward(dscores, cache_scores)
        dx, dW1, db1 = affine_relu_backward(dhidden, cache_hidden)

        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = np.random.randn(layer_dims[i], layer_dims[i+1]) * weight_scale
            self.params[f'b{i+1}'] = np.zeros(layer_dims[i+1])
            
            if self.normalization == "batchnorm" and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])

        self.dropout_param = {'mode': 'train', 'p': dropout, 'seed': seed} if self.use_dropout else None
        self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)] if self.normalization == "batchnorm" else []

        for k in self.params:
            self.params[k] = self.params[k].astype(dtype)
        
    def forward(self, X):
        cache = {}
        dropout_cache = {}
        A = X
        for i in range(1, self.num_layers):
            A, cache[f'layer{i}'] = affine_relu_forward(A, self.params[f'W{i}'], self.params[f'b{i}'])
            if self.use_dropout:
                A, dropout_cache[f'dropout{i}'] = dropout_forward(A, self.dropout_param)
        
        scores, cache[f'layer{self.num_layers}'] = affine_forward(A, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        return scores, cache, dropout_cache

     

        
    def forward(self, X):
        cache = {}
        A = X
        for i in range(1, self.num_layers):
            if self.normalization is None:
                A, cache[f'layer{i}'] = affine_relu_forward(A, self.params[f'W{i}'], self.params[f'b{i}'])
            else:
                A, cache[f'layer{i}'] = affine_norm_relu_forward(A, self.params[f'W{i}'], self.params[f'b{i}'],
                                                                 self.params[f'gamma{i}'], self.params[f'beta{i}'],
                                                                 self.bn_params[i-1])
            if self.use_dropout:
                A, cache[f'dropout{i}'] = dropout_forward(A, self.dropout_param)

        scores, cache[f'layer{self.num_layers}'] = affine_forward(A, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        return scores, cache


    def backward(self, dout, cache):
        grads = {}
        dout, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(dout, cache[f'layer{self.num_layers}'])

        for i in range(self.num_layers-1, 0, -1):
            if self.use_dropout:
                dout = dropout_backward(dout, cache[f'dropout{i}'])
            if self.normalization is None:
                dout, grads[f'W{i}'], grads[f'b{i}'] = affine_relu_backward(dout, cache[f'layer{i}'])
            else:
                dout, grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = affine_norm_relu_backward(dout, cache[f'layer{i}'])

        return grads


    def loss(self, X, y=None):
        scores, cache = self.forward(X)
        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * sum(np.sum(self.params[f'W{i}']**2) for i in range(1, self.num_layers+1))
        grads = self.backward(dout, cache)

        for i in range(1, self.num_layers+1):
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        return loss, grads

