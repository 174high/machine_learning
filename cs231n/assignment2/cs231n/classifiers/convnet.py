import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *



class FullyConnectedConvNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  [conv-relu-pool] x N - [affine-relu] x M - affine - [softmax or SVM]
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """
  def __init__(self, input_dim=(3, 32, 32), hidden_dims_affine=[100,100], num_filters=32, filter_size=7,
                num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
     - hidden_dims: A list of integers giving the size of each affine hidden layer.

    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.use_batchnorm = use_batchnorm


    C,H,W = input_dim
    F = num_filters

    self.num_conv_layers = 1
    self.num_affine_layers = len(hidden_dims_affine)
    self.num_layers = self.num_affine_layers + self.num_conv_layers

    stride_conv = 1 
    f_height = filter_size 
    f_width = filter_size
    pad = (filter_size - 1) / 2

    #Conv width height output to be used in next W2 shape
    for layer_i in range(self.num_conv_layers):
        Wi = "W{i}".format(i=layer_i+1)
        bi = "b{i}".format(i=layer_i+1)


        Cw_out = 1 + (W + 2 * pad - f_width) / stride_conv
        Ch_out = 1 + (H + 2 * pad - f_height) / stride_conv

        # Convolutional layer weights and bias
        # print Ch_out
        #w1,b1

        # first layer
        if layer_i == 0:
            self.params[Wi] = np.random.normal(scale=weight_scale, size=(F, C, f_height, f_width))
            self.params[bi] = np.zeros(F)
        else:
            print "shouldnt be hitting"
            pass
    
        # output from conv layer: [num_filters]x32x32
        # 2x2 pool will downsize: [num_filters]x16x16//double check

        p_width = 2
        p_height = 2
        stride_pool = 2 #mtypical setting, also set below in loss
        Pw_out = 1 + (Cw_out - p_width) / stride_pool
        Ph_out = 1 + (Ch_out - p_height) / stride_pool


    
    # Affine layer 1 weights and bias
    # print Cw_out
    # print F*p_width*p_height

    # old way
    # self.params['W2'] = np.random.normal(scale=weight_scale,size=(F*Pw_out*Ph_out, hidden_dim))
    # self.params['b2'] = np.zeros(hidden_dim)

    # # Affine layer 2 weights and bias
    # self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    # self.params['b3'] = np.zeros(num_classes)
    # print hidden_dim
    for layer_i in range(self.num_affine_layers):
        Wi = "W{i}".format(i=layer_i+1+self.num_conv_layers)
        bi = "b{i}".format(i=layer_i+1+self.num_conv_layers)
        if self.use_batchnorm:
            gi = "gamma{i}".format(i=layer_i+1+self.num_conv_layers)
            betai = "beta{i}".format(i=layer_i+1+self.num_conv_layers)

        if layer_i == 0: # first input
            self.params[Wi] = np.random.normal(scale=weight_scale, size=(F*Pw_out*Ph_out, hidden_dims_affine[layer_i]))
            self.params[bi] = np.zeros(hidden_dims_affine[layer_i])

            if self.use_batchnorm:
                self.params[gi] = np.ones(hidden_dims_affine[layer_i])
                self.params[betai] = np.zeros(hidden_dims_affine[layer_i])

        elif layer_i == self.num_affine_layers-1: # regular affine, no relu layer
            self.params[Wi] = np.random.normal(scale=weight_scale, size=(hidden_dims_affine[layer_i-1], num_classes))
            self.params[bi] = np.zeros(num_classes)
        else:   # affine relu layers
            self.params[Wi] = np.random.normal(scale=weight_scale, size=(hidden_dims_affine[layer_i-1], hidden_dims_affine[layer_i]))
            self.params[bi] = np.zeros(hidden_dims_affine[layer_i])

            if self.use_batchnorm:
                self.params[gi] = np.ones(hidden_dims_affine[layer_i])
                self.params[betai] = np.zeros(hidden_dims_affine[layer_i])


    print self.params.keys()

    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_affine_layers - 1)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # W1, b1 = self.params['W1'], self.params['b1']
    cache = {}  # storing our computations for ease of use in backprop
                # originally used self.params, but reserve this only for
                # Wi and bi.  Interferes with automated checks
    
    # pass conv_param to the forward pass for the convolutional layer
    # filter_size = W1.shape[2]
    # print self.params['W1'].shape
    filter_size = self.params['W1'].shape[2] #XXX: double check this, will this change from 7?

    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #conv - relu - 2x2 max pool - [affine - relu] x M - affine - softmax


    for layer_i in range(self.num_conv_layers):
        Wi = "W{i}".format(i=layer_i+1)
        bi = "b{i}".format(i=layer_i+1)
        ci = "c{i}".format(i=layer_i+1) 
        # print ci
        W = self.params[Wi]
        b = self.params[bi]

        scores, cache[ci] = conv_relu_pool_forward(X, W, b, conv_param, pool_param)

    #(50, 32, 16, 16)
    print X.shape
    print scores.shape

    # Affine xxx
    for layer_i in range(self.num_affine_layers):
        Wi = "W{i}".format(i=layer_i+1+self.num_conv_layers)
        bi = "b{i}".format(i=layer_i+1+self.num_conv_layers)
        ci = "c{i}".format(i=layer_i+1+self.num_conv_layers) 
        BNi = "BNC{i}".format(i=layer_i+1+self.num_conv_layers)   # BatchNorm cache
        
        gi = "gamma{i}".format(i=layer_i+1+self.num_conv_layers)
        betai = "beta{i}".format(i=layer_i+1+self.num_conv_layers)

        W = self.params[Wi]
        b = self.params[bi]

        if layer_i == self.num_affine_layers-1: # output layer
            scores, cache[ci] = affine_forward(scores, W, b)
        else:
            scores, cache[ci] = affine_relu_forward(scores, W, b)
            if self.use_batchnorm:
                gamma = self.params[gi]
                beta = self.params[betai]
                scores, bn_cache = batchnorm_forward(scores, gamma, beta, self.bn_params[layer_i])
                cache[BNi] = bn_cache

    # scores = out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # print scores
    loss, dx = softmax_loss(scores, y)

    regularization = 0
    for layer_i in range(self.num_layers):
        Wi = "W{i}".format(i=layer_i+1)
        W = self.params[Wi]
        regularization += (0.5 * self.reg * np.sum(W**2))

    loss += regularization

    #softmax - affine - relu - affine - 2x2 max pool - relu - conv
    for layer_i in range(self.num_layers-1, self.num_conv_layers-1, -1):
        Wi = "W{i}".format(i=layer_i+1)
        bi = "b{i}".format(i=layer_i+1)
        ci = "c{i}".format(i=layer_i+1) 

        BNi = "BNC{i}".format(i=layer_i+1)        # Batchnorm cache
        gammai = "gamma{i}".format(i=layer_i+1)   # gamma gradient
        betai = "beta{i}".format(i=layer_i+1)     # beta gradient

        if layer_i == self.num_affine_layers:
            dx,dw,db = affine_backward(dx, cache[ci])
        else:
            if self.use_batchnorm:
                bn_cache = cache[BNi]
                dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
                grads[gammai] = dgamma
                grads[betai] = dbeta


            dx,dw,db = affine_relu_backward(dx, cache[ci])


        grads[Wi] = dw + self.reg * self.params[Wi]
        grads[bi] = db


    for layer_i in range(self.num_conv_layers-1, -1, -1):
        Wi = "W{i}".format(i=layer_i+1)
        bi = "b{i}".format(i=layer_i+1)
        ci = "c{i}".format(i=layer_i+1) 

        dx,dw,db = conv_relu_pool_backward(dx, cache[ci])
        grads[Wi] = dw + self.reg * self.params[Wi]
        grads[bi] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
pass
