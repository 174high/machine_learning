import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = np.dot(X[i], W) # one row of scores (10,)
    scores -= np.max(scores)

    exp_scores = np.exp(scores)
    p = exp_scores / np.sum(exp_scores)
    loss += -np.log(p[y[i]])

    for j in range(num_classes):
      #http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
      #p_i (omit the  -y_i in the stack exchange)
      dW[:,j] += np.exp(scores[j]) / np.sum(exp_scores) * X[i,:] # deriv of lost function
      if j == y[i]:
        dW[:,j] -= X[i,:] 

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = np.dot(X, W)
  scores -= np.max(scores)

  exp_scores = np.exp(scores)
  p = exp_scores/np.sum(exp_scores)

  norm_scores = (exp_scores.T/np.sum(exp_scores, axis=1)).T

  loss = np.sum(-np.log(norm_scores[range(num_train), y]))
  loss = loss / num_train + 0.5 * reg * np.sum(W*W)


  """ 
  Gradient
  First set dW base as X.T * norm_scores

  Then create mask that will subtract each correct class by X[i,:]
  """
  dW =  np.dot(X.T, norm_scores)
  mask = np.zeros(norm_scores.shape)

  # Subtract each row's correct class by X[i,:]
  mask[range(X.shape[0]),y] = -1
  cor_scores = np.dot(X.T, mask)
  dW += cor_scores        

  dW = dW / num_train + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

