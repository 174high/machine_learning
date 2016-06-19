import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  p_state=0
  c_state=0

  # h_t = f_w(prev_h, x)

  # previous state
  # N,H * H,H -> N,H
  p_state = np.dot(prev_h, Wh)

  # current state
  # N,D * D,H -> N,H
  c_state = np.dot(x, Wx)

  next_h = np.tanh(p_state + c_state + b)
  # Order is important here, x needs to be first, I think on last iteration
  #   one of these is none, and doesn't get added to tuple
  cache = x, p_state, c_state, Wx, prev_h, Wh, b

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x, p_state, c_state, Wx, prev_h, Wh, b) = cache
  # backprop into next_h = np.tanh(p_state + c_state + b)
  df = (1-np.tanh(p_state+c_state+b)**2) * dnext_h 
  dp_state = df            #(N, H)
  dc_state = df            #(N, H)
  db = np.sum(df, axis=0)  #(H,)

  # backprop into c_state = np.dot(x, Wx)
  dx = np.dot(dc_state, Wx.T)
  dWx = np.dot(x.T, dc_state)

  # backprop into p_state = np.dot(prev_h, Wh)
  dprev_h = np.dot(dp_state, Wh.T)
  dWh = np.dot(prev_h.T, dp_state)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################

  # (N, T, D)
  # rnn_step_forward(x, prev_h, Wx, Wh, b)

  N,T,D = x.shape
  _,H = h0.shape
  h = np.empty((N,T,H))
  cache = [] 

  for t in range(x.shape[1]):
    if t==0:
      h_prev = h0

    h_prev, cnext = rnn_step_forward(x[:,t,:], h_prev, Wx, Wh, b)
    h[:,t,:] = h_prev
    cache.append(cnext)

  # output (N, T, H)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################

  # rnn_step_backwards takes in (N, D)
  # dh has shape (N, T, H), we loop over T
  N,T,H = dh.shape

  D = cache[0][0].shape[1] # X dimension (N, D), X is 0 element
  # D = 3

  dx = np.empty((T,N,D))
  dWh = np.zeros((H, H))
  dWx = np.zeros((D, H))
  dh_prev = np.zeros((N, H))
  dh0 = np.zeros((N, H))
  db = np.zeros((H))
  
  # Reverse upstream of gradients to get the last grad (from fwd pass) first
  for i in reversed(range(T)):
    dh_current = dh[:,i,:] + dh_prev
    dx_t, dh_prev, dWx_t, dWh_t, db_t = rnn_step_backward(dh_current, cache[i])
    dx[i] = dx_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t
    if i==0:
      dh0 = dh_prev

  # Want N,T,D from T,N,D
  dx  = dx.transpose(1,0,2)
  # dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  # x: (N, T), minibatch of size N, each batch has length T

  # Old method:

  # N,T = x.shape
  # V,D = W.shape

  # c_batch = np.empty((T,V))
  # out = np.empty((N,T,D))
  # for batch_i in range(N):
  #   for i, word in enumerate(x[batch_i]):
  #     converted = np.zeros(V)
  #     converted[word] = 1 
  #     c_batch[i] = converted

  #   out[batch_i] = np.dot(c_batch, W)

  out = W[x]
  cache = (x,W)


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  dW = np.zeros(W.shape)
  np.add.at(dW, x, dout) # x = indexes to add dout (gradient) to 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
    

  # Step 1 compute activation vector 
  activation_vector = np.dot(x, Wx) + np.dot(prev_h, Wh) + b 

  # Step 2, divide activation vector into four vectors ai, af, ao, ag and 
  # then compute the gates
  # activation vector (3,20), need to divide by 4 to give i,f,o,g each 5 for (3,5)
  v_size = activation_vector.shape[1]/4


  ai = activation_vector[:,0*v_size:0*v_size+(v_size)]
  af = activation_vector[:,1*v_size:1*v_size+(v_size)]
  ao = activation_vector[:,2*v_size:2*v_size+(v_size)]
  ag = activation_vector[:,3*v_size:3*v_size+(v_size)]

  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)

  # Step 3 compute the next cell state
  next_c = f * prev_c + i * g

  # Step 4 compute next hidden state
  next_h = o * np.tanh(next_c)

  cache = (i,f,o,g,ai,af,ao,ag, prev_h, Wh, x, Wx, next_c, prev_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  (i,f,o,g,ai,af,ao,ag, prev_h, Wh, x, Wx, next_c, prev_c) = cache

  # 4 backprop into next_h = o * np.tanh(next_c)
  do = np.tanh(next_c) * dnext_h
  dnext_c += (1 - (np.tanh(next_c)**2)) * o * dnext_h

  # 3 backprop into next_c = f * prev_c + i * g
  df = prev_c * dnext_c
  dprev_c = f * dnext_c
  di = g * dnext_c
  dg = i * dnext_c

  # 2 back prop into activation layer
  # i = sigmoid(ai)
  # f = sigmoid(af)
  # o = sigmoid(ao)
  # g = np.tanh(ag)
  dag = (1-g**2) * dg
  dao = o*(1-o) * do
  daf = f*(1-f) * df
  dai = i*(1-i) * di

  # Stack each of the activation vector derivatives to get complete derivative of
  # activation vector
  dactivation_vector = np.hstack((dai,daf,dao,dag))

  # backprop into activation_vector = np.dot(x, Wx) + np.dot(prev_h, Wh) + b 
  dx = np.dot(dactivation_vector, Wx.T)
  dWx = np.dot(x.T,dactivation_vector)

  dprev_h = np.dot(dactivation_vector, Wh.T)
  dWh = np.dot(prev_h.T, dactivation_vector)
  db = np.sum(dactivation_vector, axis=0) 

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  
  N,T,D = x.shape
  _,H = h0.shape
  # cell shape (N,H)
  c0 = np.zeros((N,H))
  h = np.zeros((N,T,H))
  cache = []
  for t in xrange(T):
    if t == 0:
      prev_c = c0
      prev_h = h0

    # lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
    prev_h, prev_c, cache_t = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)
    cache.append(cache_t)
    h[:,t,:] = prev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

