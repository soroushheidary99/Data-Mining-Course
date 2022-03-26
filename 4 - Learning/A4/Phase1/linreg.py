import numpy as np


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100, momentum=0.95):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
        self.h = None
        self.momentum = momentum

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
        self.JHist = []
        for i in range(self.n_iter):
            self.JHist.append((self.computeCost(X, y, theta), theta))
            cost = self.JHist[i][0]
            h = self.predict(X)
            print("Iteration: ", i+1, " Cost: ", cost, " Theta: ", theta)
            for j in range(0, d):
                diff = np.subtract(h, y.reshape(n))
                all_coeffs = X.transpose()[j]
                mull = np.multiply(diff, all_coeffs)
                theta[j] = theta[j] - (2 * self.alpha/n) *  np.sum(mull)
            self.alpha *= self.momentum
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
              ** make certain you don't return a matrix with just one value! **
        '''
        pred = np.dot(X,theta)
        cost = np.sqrt(np.mean(np.square(np.subtract(pred,y))))

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
        h = np.ones((X.shape[0],1))
        for i in range(0, X.shape[0]):
            mul = np.matmul(self.theta, X[i])
            h[i] = float(np.sum(mul[0]))
        h = h.reshape(X.shape[0])
        return h
