import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']
    num_train = X.shape[0]

    X = np.reshape(X, (X.shape[0], -1))

    # (N, D) . (D, H) -> (N, H) (input, hidden dim)
    L1_scores = np.dot(X, W1) + b1
    L1 = np.maximum(0, L1_scores)
    scores = np.dot(L1, W2) + b2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
      
    #
    # blah = [[1,2,3], [1,2,3]]
    # np.sum(blah, axis=1, keepdims=True)
    # array([[6],
    #        [6]])
    # Using softmax:
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    regs = (0.5 * self.reg * np.sum(W1*W1)) + (0.5 * self.reg * np.sum(W2*W2)) 
    loss = np.sum(-np.log(p[range(num_train), y])) / num_train + regs


    dScores = p
    dScores[range(num_train), y] -= 1
    dScores /= num_train

    # backprop into scores = np.dot(L1, W2) + b2
    dB2 = (1) * np.sum(dScores, axis=0)

    """
    XXX: Very important, order matters:
    dL1 = np.dot(dScores, W2.T)
    dW2 = np.dot(dScores.T, L1)
    produces valid dims for dot prod, but wrong results (7, 50)

    dW2 needs to be same dim as W2 dimension: (50, 7) (H, C)

    """
    dL1 = np.dot(dScores, W2.T)
    dW2 = np.dot(L1.T, dScores)

    # back prop into L1 = np.maximum(0, L1_scores)
    dL1_scores = L1_scores
    dL1_scores[dL1_scores>0] = 1
    dL1_scores[dL1_scores<=0] = 0
    dL1_scores *= dL1

    # backprop into L1_scores = np.dot(X, W1) + b1
    dB1 = (1) * np.sum(dL1_scores, axis=0)
    dX = np.dot(dL1_scores, W1.T)
    dW1 = np.dot(X.T, dL1_scores)


    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['b1'] = dB1
    grads['b2'] = dB2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    for layer_i in range(self.num_layers-1):
        Wi = "W{i}".format(i=layer_i+1)
        bi = "b{i}".format(i=layer_i+1)

        if layer_i == 0:
            self.params[Wi] = weight_scale * np.random.randn(input_dim, hidden_dims[layer_i])
            self.params[bi] = np.zeros(hidden_dims[layer_i])
        else:
            self.params[Wi] = weight_scale * np.random.randn(hidden_dims[layer_i-1], hidden_dims[layer_i])
            self.params[bi] = np.zeros(hidden_dims[layer_i])

    if self.use_batchnorm:
        for layer_i in range(self.num_layers-2):
            gi = "gamma{i}".format(i=layer_i+1)
            bi = "beta{i}".format(i=layer_i+1)
            self.params[gi] = np.ones(hidden_dims[layer_i])
            self.params[bi] = np.zeros(hidden_dims[layer_i])


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # for layer_i in range(self.num_layers-2):
    #     self.bn_params[layer_i]['running_mean'] = np.zeros(hidden_dims[layer_i])
    #     self.bn_params[layer_i]['running_var'] = np.zeros(hidden_dims[layer_i])
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    # Unroll input dimenions
    # ex (25,3,32,32) -> (25, 3072)
    X = np.reshape(X, (X.shape[0], -1)) 
    num_train = X.shape[0]
    cache = {}  # storing our computations for ease of use in backprop
                # originally used self.params, but reserve this only for
                # Wi and bi.  Interferes with automated checks

    for layer_i in range(self.num_layers)[1:]:
        Wi = "W{i}".format(i=layer_i)
        bi = "b{i}".format(i=layer_i)
        Ai = "A{i}".format(i=layer_i)       # Affine cache
        Ri = "R{i}".format(i=layer_i)       # ReLU cache
        BNCi = "BNC{i}".format(i=layer_i)   # BatchNorm cache

        _W = self.params[Wi]
        _b = self.params[bi]

        # First layer, input X
        if layer_i == 1: 
            scores, a_cache = affine_forward(X, _W, _b)
        else:
            scores, a_cache = affine_forward(scores, _W, _b)

        cache[Ai] = a_cache

        # Apply ReLU & Batchnorm only to non-output layers
        if layer_i < self.num_layers-1:
            if self.use_batchnorm:
                gi = "gamma{i}".format(i=layer_i)
                bi = "beta{i}".format(i=layer_i)
                gamma = self.params[gi]
                beta = self.params[bi]
                scores, bn_cache = batchnorm_forward(scores, gamma, beta, self.bn_params[layer_i-1])
                # cache[BNi] = scores
                cache[BNCi] = bn_cache

            scores, r_cache = relu_forward(scores)
            cache[Ri] = r_cache


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    # Compute regularization:
    regs = 0
    for layer_i in range(self.num_layers-1):
        Wi = "W{i}".format(i=layer_i+1)
        _W = self.params[Wi]
        regs += (0.5 * self.reg * np.sum(_W*_W))

    loss, dScores = softmax_loss(scores, y)
    loss += regs

    for layer_i in range(self.num_layers-1,0,-1):
        Wi = "W{i}".format(i=layer_i)   # W gradient
        bi = "b{i}".format(i=layer_i)   # bias gradient
        Ai = "A{i}".format(i=layer_i)   # Affine cache
        Ri = "R{i}".format(i=layer_i)   # ReLU cache

        gammai = "gamma{i}".format(i=layer_i)   # gamma gradient
        betai = "beta{i}".format(i=layer_i)     # beta gradient

        # batchnorm:
        # BNi = "BN{i}".format(i=layer_i)
        BNCi = "BNC{i}".format(i=layer_i)

        # back prop into ReLU L1 = np.maximum(0, L1_scores)
        # only applied on hidden layers:
        # print str(layer_i)  + "," + str(self.num_layers-1)
        if layer_i < self.num_layers-1: 
            r_cache = cache[Ri]
            dScores = relu_backward(dScores, r_cache)

            if self.use_batchnorm:
                bn_cache = cache[BNCi]
                dScores, dgamma, dbeta = batchnorm_backward(dScores, bn_cache)
                
                grads[gammai] = dgamma
                grads[betai] = dbeta


        a_cache = cache[Ai]
        _,W,_ = a_cache
        dScores,dW,db = affine_backward(dScores, a_cache)
        grads[bi] = db
        grads[Wi] = dW + self.reg * W

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

