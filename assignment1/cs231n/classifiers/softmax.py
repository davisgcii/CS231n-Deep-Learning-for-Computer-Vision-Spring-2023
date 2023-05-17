from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros_like(W) # ()

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    verbose = False

    for i in range(num_train):
      scores = np.dot(X[i], W) # shape: (1, C)

      # normalize the scores to prevent overflows and NaNs
      scores -= np.max(scores)

      # get all probabilities; useful for dW calcs
      probabilities = np.exp(scores) / np.sum(np.exp(scores)) # (1, C)
      

      loss += -np.log(probabilities[y[i]])

      # now get dscores, or dL/dprobs: dscores = probs - 1(where score=y[i])
      dscores = probabilities # (1, C)
      dscores[y[i]] -= 1

      # X[i] is (1, D)
      dW += np.outer(X[i], dscores) # (D, C)
        
    # same as SVM
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    if verbose:
        print(f"scores: {scores.shape}")
        print(f"propabilities: {probabilities.shape}")
        print(f"dscores: {dscores.shape}")
        print(f"dW: {dW.shape}")
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = np.dot(X, W) # (N, C)

    # prevent overflows by subtracting max value from each row
    scores -= scores.max(axis=1, keepdims=True) 

    # calculate the probabilities; (N, C)
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # calculate the loss
    losses = -np.log(probabilities[np.arange(num_train), y]) # (N, 1)
    loss += np.sum(losses) / num_train + reg * np.sum(W * W)
    
    # get dscores; subtract 1 from each row's correct class score
    dscores = probabilities # (N, C)
    dscores[np.arange(num_train), y] -= 1

    dW = np.dot(X.T, dscores) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
