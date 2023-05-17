from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # scores is shape (1, C)
        correct_class_score = scores[y[i]] # this is shape (1,1)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += -X[i].T # for when j==y[i]
                dW[:,j] += X[i].T # for when j!=y[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # divide by num_train to give an average gradient
    dW /= num_train
    
    # add regularization component to the gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

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
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero; shape is (D, C)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # D is number of dimentions
    # C is number of classes
    # N is number of training rows
    # loss is a single float

    scores = X.dot(W) # scores is shape (N, C)
    delta = 1
    N = X.shape[0]
    C = W.shape[1]

    # need to get the correct-class scores (i.e., the score predicted for class y[i] for each x[i])
    correct_scores = scores[np.arange(N), y].reshape(-1,1)
    
    # now get all of the margins by subtracting the correct scores from the scores
    # use np.maximum to get the element-wise max
    margins = np.maximum(0, scores - correct_scores + delta) # shape (N, C)

    # we don't want to sum up the margins where j==y[i] so set those values to zero
    margins[np.arange(N), y] = 0

    # sum up the loss, divide by N, and add regularization
    loss += (np.sum(margins) / N + reg * np.sum(W * W)) # a scalar

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # identify the margins that contribute to the loss (i.e., are >0)
    loss_contributors = np.zeros(margins.shape)
    loss_contributors[margins > 0] = 1 # shape of (N, C), same as margins

    # sum up the classes that contribute to the loss and insert into class y[i]
    loss_contributors[np.arange(N), y] = -np.sum(loss_contributors, axis=1)

    dW = X.T.dot(loss_contributors) # gives shape (D, C)

    # divide by N and add regularization
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
