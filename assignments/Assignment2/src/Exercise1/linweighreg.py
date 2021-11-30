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
            
    def fit(self, X, t):
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

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        #compute A
        A = numpy.diag(t.T[0] ** 2) #
        # compute weights
        a = numpy.dot(X.T, numpy.dot(A,X)) #left side
        b = numpy.dot(X.T, numpy.dot(A,t)) #right side

        self.w = numpy.linalg.solve(a, b)
        
        # (2) TODO: Make use of numpy.linalg.solve instead!
        # Reason: Inverting the matrix is not very stable
        # from a numerical perspective; directly solving
        # the linear system of equations is usually better.

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        # (1) TODO: Compute the predictions for the
        # array of input points

        X = numpy.array(X).reshape((len(X), -1))

        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        predictions = numpy.dot(X, self.w)
        #ones = numpy.ones((predictions.shape[0], 1))
        #predictions = numpy.concatenate((ones, predictions), axis=1)
        return predictions