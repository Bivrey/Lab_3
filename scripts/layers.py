from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Выполняет прямой проход для аффинного (полносвязного) слоя.

    x: входные данные размера (N, d_1, ..., d_k)
    w: веса размера (D, M), где D = np.prod(d_1, ..., d_k)
    b: смещения размера (M,)

    Возвращает:
    out: выходные данные размера (N, M)
    cache: кортеж переменных (x, w, b) для использования в обратном проходе
    """
    N = x.shape[0]
    x_row = x.reshape(N, -1)  
    out = np.dot(x_row, w) + b  
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Вычисляет обратный проход для аффинного слоя.
    
    Входы:
    - dout: Производная потерь по выходу слоя, форма (N, M)
    - cache: Кортеж, содержащий:
      - x: Входные данные, форма (N, d_1, ... d_k)
      - w: Веса, форма (D, M)
      - b: Смещения, форма (M,)
    
    Возвращает кортеж:
    - dx: Градиент по входным данным x, форма (N, d1, ..., d_k)
    - dw: Градиент по весам w, форма (D, M)
    - db: Градиент по смещениям b, форма (M,)
    """
    x, w, b = cache
    
    x_reshaped = x.reshape(x.shape[0], -1)  
    
    db = np.sum(dout, axis=0)  
    
    dw = np.dot(x_reshaped.T, dout)  
    
    dx = np.dot(dout, w.T) 
    dx = dx.reshape(*x.shape)  
    
    return dx, dw, db

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform followed by batch normalization and a ReLU

    Inputs:
    - x: Input data of shape (N, D)
    - w, b: Weights and biases for the affine layer
    - gamma, beta: Parameters for batch normalization
    - bn_param: Dictionary with parameters for batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Tuple of values needed for the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_norm_relu_backward(dout, cache):
        """
        Backward pass for the affine-batchnorm-relu convenience layer
        """
        fc_cache, bn_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        da_bn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
        dx, dw, db = affine_backward(da_bn, fc_cache)
        return dx, dw, db, dgamma, dbeta
    
    
def relu_forward(x):
    """
    Вычисляет прямой проход для слоя с активацией ReLU.

    Вход:
    - x: Входные данные любой формы

    Возвращает кортеж:
    - out: Выходные данные, той же формы, что и x
    - cache: кэш содержащий входные данные x для использования в обратном проходе
    """
    out = np.maximum(0, x)  
    cache = x  
    return out, cache


def relu_backward(dout, cache):
    """
    Вычисляет обратный проход для слоя ReLU.

    Входы:
    - dout: Производная потерь по выходу ReLU, любой формы
    - cache: Кэш, содержащий входные данные x, использованные в relu_forward

    Возвращает:
    - dx: Градиент потерь по входу x
    """
    x = cache
    dx = dout * (x > 0)  
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        
        out = gamma * x_normalized + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, x_normalized, gamma, beta, sample_mean, sample_var, eps)

    elif mode == "test":
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

        out = gamma * x_normalized + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    """
    x, x_normalized, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)

    dx_normalized = dout * gamma

    x_mu = x - sample_mean
    std_inv = 1. / np.sqrt(sample_var + eps)
    dvar = np.sum(dx_normalized * x_mu, axis=0) * -0.5 * std_inv**3

    dmu = np.sum(dx_normalized * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

    dx = (dx_normalized * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        # Create the dropout mask and apply it to the data
        mask = (np.random.rand(*x.shape) < p) / p  # Scale activations during training
        out = x * mask  # Apply dropout mask
    elif mode == 'test':
        # During testing, we do not drop or scale any units
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        # Only back-propagate the gradients where the dropout mask is non-zero
        dx = dout * mask
    elif mode == 'test':
        # In test mode, simply pass through all gradients
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for n in range(N):  
        for f in range(F):  
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + HH
                    w_end = w_start + WW
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out, W_out = dout.shape[2], dout.shape[3]
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + HH
                    w_end = w_start + WW
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]
                    dw[f] += dout[n, f, i, j] * x_slice
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += dout[n, f, i, j] * w[f]

    dx = dx_padded[:, :, pad:-pad, pad:-pad]  
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h1 = h * stride
                    w1 = w * stride
                    h2 = h1 + pool_height
                    w2 = w1 + pool_width
                    window = x[n, c, h1:h2, w1:w2]
                    out[n, c, h, w] = np.max(window)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H_out, W_out = dout.shape
    H, W = x.shape[2], x.shape[3]

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h1 = h * stride
                    w1 = w * stride
                    h2 = h1 + pool_height
                    w2 = w1 + pool_width
                    window = x[n, c, h1:h2, w1:w2]
                    max_val = np.max(window)
                    for i in range(pool_height):
                        for j in range(pool_width):
                            if window[i, j] == max_val:
                                dx[n, c, h1 + i, w1 + j] += dout[n, c, h, w]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
