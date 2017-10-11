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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num = np.shape(X)[0]
  num_class = np.shape(W)[1]
  total = np.zeros((num,1))
  for i in range(num):
      for j in range(num_class):
          total[i,0] = total[i,0] + np.exp(np.sum(np.multiply(X[i,:],W[:,j])))
      loss = loss - np.log( np.exp(np.sum(np.multiply(X[i,:],W[:,y[i]]))) / total[i,0])
  loss = loss/num
  loss = loss + 0.5*reg*np.sum(W**2)
  
  ftemp = np.zeros((num, num_class))
  for i in range(num):
      for j in range(num_class):
         ftemp[i,j] =  np.exp(np.sum(np.multiply(X[i,:],W[:,j])))/total[i,0]
         if y[i] == j:
             ftemp[i,j] = ftemp[i,j]-1

  dW = np.asmatrix(X.T)*np.asmatrix(ftemp)/num
  dW = dW + reg*W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num = np.shape(X)[0]
  num_class = np.shape(W)[1]
  temp = np.dot(X, W) #n*c

  row_max = np.max(temp, axis = 1,keepdims = True)
  temp = temp - row_max
  temp = np.exp(temp)
  
  summat = np.sum(temp,axis = 1,keepdims=True)  #N*1
  real = temp[np.arange(num),y]
  loss = np.sum(-np.log(real[:,np.newaxis]/summat))#500*????q
  loss = loss/num
  loss = loss + 0.5*reg*np.sum(W**2)
  
  f = np.zeros((num,num_class))
  for i in range(num):
     # print(np.shape(f),np.shape(temp),summat[i])
      f[i,:] = temp[i,:]/summat[i]
      f[i,y[i]] = f[i,y[i]] - 1
  gradient = np.dot(X.T,f)/num
  gradient = gradient + reg*W
  dW = gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

