import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)

    correct_class_score = scores[y[i]]
    """ 
    Was going to count and save index, but don't need.  The continue
    statement is effectively saving in the sense that we want to 
    count the number of classes that did NOT meet the criteria, which
    ends up running in parallel with j!=y.  

    # count = 0       
    # saved_index = 0 
    """
    for j in xrange(num_classes):
      if j == y[i]:
        # saved_index = j  # Don't need to save, continue skips 
        # Skip because we are counting loss when it's not the right class
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:  # max(0, s_j-s_y+delta)
        loss += margin
        dW[:,j] += X[i,:]     # j != y
        dW[:,y[i]] -= X[i,:]  # j == y  # contributing to loss function
                              # important to note, y[i], so it will sums compared
                              # to above dW[:,j] which runs only once per class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train
  dW += reg*W

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  """
  Loss:
  Subtract loss if 

  Gradient:  
  http://cs231n.github.io/optimization-1/


  dW_y_i = -(1)*count*x_i IF margin > 0 and count = number of classes where j=y
  dW_j = 1*count*x_i IF margin > 0 and count = number of classes where j!=y 

  Both run the same amount of times.  dW_y subtracts sum of total x[i,:] only with 
  respect to the row of W that corresponds to the correct class that doesn't 
  meet criteria of j==y[i] and margin > 0. 

  dw_j runs where j!=y[i] and margin > 0 

  """

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y]
  margins = np.maximum(0,(scores - np.array([correct_class_scores]).T + 1.0))

  # Omit the correct class score because it does not contribute to loss:
  margins[np.arange(num_train), y] = 0

  loss = margins.sum()/len(margins)
  loss += .05 * reg * np.sum(W*W)

  # Original scratchpad attempt:  
  # delta = 1
  # for j in xrange(num_classes):
  #   filtered_class = np.array(scores)[np.where(y==j)] # get all rows where class = 0
  #   correct_scores = filtered_class[:,j]              # get column of correct scores
  #   incorrect_scores = np.delete(filtered_class, j, axis=1) # get incorrect columns
  #   incorrect_scores_row_sums = np.sum(incorrect_scores, axis=1)
  #   margin_scores = ((incorrect_scores_row_sums - correct_scores)+1)
  #   margin_scores[margin_scores<0]=0
  #   correct_scores_rep = np.tile(np.array([correct_scores]).transpose(),(1,9))
  #   test = (incorrect_scores - correct_scores_rep + 1)
  #   test[test<0] = 0
  #   loss += np.sum(test)/len(test)
  #   # test = ((incorrect_scores_row_sums - correct_scores*9)+1)
  #   # test[test<0] = 0
  #   # loss+=np.sum(test)/len(test)
  #   # loss += (np.sum(margin_scores)/len(margin_scores))
  #   loss += 0.5 * reg * np.sum(W * W)
  #   print loss

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  """
  Create matrix that will be dot product with X in order to sum.

  Count each row where # of margins > 1.  This is for -1 * summation 
  where the row corresponds to the correct class.  Others will be set to 1

  Create matrix of ones, set to one where margins > 0.  
  Set row,col = margins_gt_one, which is 

  ex of this:
  test=np.array([[1,2,3,4,5],[11,22,33,44,55],[111,222,333,444,555]])
  test[np.arange(3), np.arange(3)]
  test[np.arange(3), np.arange(3)] = [-1,-1,-1]
  test

  """
  # Get total count of positives per row
  total_positives = np.sum(margins>0, axis=1) # 500x10 -> 500,
  mask = np.zeros(margins.shape)
  mask[margins>0] = 1

  # Set list of x_i,y_i where each x_i is correct class, which is total of 500.  
  # Set to the list of total positives vector (dim 500)
  # 500x10
  mask[np.arange(num_train), y] = -total_positives 

  dW = np.dot(mask.T, X)
  dW /= num_train
  dW = dW.T
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
