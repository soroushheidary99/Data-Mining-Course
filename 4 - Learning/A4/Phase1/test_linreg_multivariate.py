import numpy as np

from linreg import LinearRegression


if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    
    # load the data
    filePath = 'data/multivariateData.dat'
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # # Standardizing y
    # mean = y.mean(axis=0)
    # std = y.std(axis=0)
    # y = (y - mean) / std

    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 100
    alpha = 0.007

    print('THETA:', init_theta)
    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter, momentum=0.95)
    lr_model.fit(X,y)
    

    print('-'*50)
    test_path = r'data/holdout.npz'

    test = np.load(test_path)['arr_0']
    X_test = np.matrix(test[:,:-1])
    y_test = np.matrix(test[:,-1]).T


    mean = X_test.mean(axis=0)
    std = X_test.std(axis=0)
    X_test = (X_test - mean) / std
    print('The RMSE Loss For The Test Data: ', end='')
    print(np.sqrt(np.mean(np.square(np.subtract(lr_model.predict(X_test), y_test)))))
