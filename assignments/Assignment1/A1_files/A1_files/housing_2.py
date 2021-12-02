import numpy as np
import pandas as pd
import linreg 
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print("Single model weight: %a\n"% model_single.w)

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
#for-loop to make more readable output, simply type print(model_all.w) if the 
#look of the output doesn't matter
for i in range(len(model_all.w)): 
    print("All model weight w%a: %a"% (i, model_all.w[i][0]))

# (d) evaluation of results
def rmse(t, tp):
    error = t - tp
    square = error ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)       
    return root

print("RMSE of all: %a"% rmse(t_test, model_all.predict(X_test)))

print("RMSE of single: %a"% rmse(t_test, model_single.predict(X_test[:,0])))

plt.subplot(212)
plt.title("2D scatter plot(\"true house prices\" vs. \"estimates\") - ALL")
plt.plot([10,20,30,40,50],[10,20,30,40,50])
plt.scatter(model_all.predict(X_test), t_test, color='cyan', label="All")
plt.ylabel('true house prices')
plt.legend(loc=2)
plt.xlim(-18,45)
plt.savefig("2DscatterPlotModelALL.png")


plt.subplot(211)
plt.title("2D scatter plot(\"true house prices\" vs. \"estimates\") - SINGLE")
plt.scatter(model_single.predict(X_test[:,0]), t_test, color='purple', label="Single")
plt.ylabel('true house prices')
plt.xlabel('Predictions')
plt.legend(loc=2)
plt.xticks([-18,0,10,20,30,40,45])
ax = plt.gca()
ax.xaxis.set_visible(False)
plt.savefig("2DscatterPlotModelSingle.png")
