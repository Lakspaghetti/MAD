from matplotlib.pyplot import axes
import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
    #        
    def fit(self, X, t, Lam, k):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        

        # make sure that we have Numpy arrays; also
        # reshape the target array to ensure that we have
        # a N-dimensional Numpy array (ndarray), see
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        constx = X #copy X, so we can alter it without losing its values
        #k'th order matrix to get k'th order polynomial
        if (k>1):
            X = X ** 0 #0'th order polynomial, column of ones
            for i in range(1,k+1):
                X = numpy.concatenate((X, constx ** i), axis=1)
        else:
            ones = numpy.ones((X.shape[0], 1))
            X = numpy.concatenate((ones, X), axis=1)

        #create identity matrix
        I = numpy.eye(len(X[0]))

        # compute weights
        a = numpy.dot(X.T,X) + (len(X) * Lam * I)  #X.T is X transposed
        b = numpy.dot(X.T,t)
        self.w = numpy.linalg.solve(a, b)

    #
    def predict(self, X, k):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        X = numpy.array(X).reshape((len(X), -1))

        constx = X
        if (k>1):
            X = X ** 0 #0'th order polynomial, column of ones
            for i in range(1,k+1):
                X = numpy.concatenate((X, constx ** i), axis=1)
        else:
            ones = numpy.ones((X.shape[0], 1))
            X = numpy.concatenate((ones, X), axis=1)

        predictions = numpy.dot(X, self.w)

        return predictions    
    
    #
    def LOOCV(self, x, t, lamValue, k):
        x = numpy.array(x).reshape((len(x), -1))
        t = numpy.array(t).reshape((len(t), 1))

        errors = []
        

        for n in range(len(x)):
            self.fit(numpy.delete(x,n), numpy.delete(t,n), lamValue, k)#find w_{-n}
            error = t[n] - self.predict(x[n], k)
            errors = numpy.append(errors, error)
        
        square = errors ** 2
        mean = numpy.mean(square)
        return mean