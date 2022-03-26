import numpy as np
import math     
import time

class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        factor = 0.9
        for i in range(self.n_iter):
            c = self.computeCost(X, y, theta)
            time.sleep(0.3)
            # self.JHist.ap/pend((c,theta))
            # predictions = np.dot(X, theta)
            # j_function = np.dot(X.T, predictions-y)/(2*n)
            a = np.subtract(np.dot(X,theta),y)
            b = np.dot(a.T,a)
            theta -= self.alpha*b/(2*n)
            self.alpha*=factor
            print("Iteration: ", i+1, " Cost: ", c)
            
        return theta

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
               make certain you don't return a matrix with just one value! 
        '''
        # print(np.mean(np.subtract(np.dot(X,theta),y)))
        # print(X,y,theta)
        now = np.dot(X,theta)
        cost = np.sqrt(np.mean(np.subtract(now,y)))
        return cost

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        return np.dot(X, self.theta)