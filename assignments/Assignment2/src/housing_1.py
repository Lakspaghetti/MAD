import numpy as np
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1)) #house prices
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set - 
Avg_price = sum(t_train)/len(t_train) #numpy.mean #22.01660079


# (b) RMSE function
def rmse(t, tp):
    error = t - tp
    square = error ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)       
    return root

estimates = np.full((len(t_test), 1), Avg_price)
rmse(t_test, estimates)

# (c) visualization of results
plt.title("2D scatter plot(\"true house prices\" vs. \"estimates\")")
plt.scatter(estimates, t_test)
plt.ylabel('true house prices')
plt.xlabel('estimates')
plt.savefig("2DscatterPlot.png")
