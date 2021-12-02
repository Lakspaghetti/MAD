import numpy as np
import pandas as pd
import linweighreg  as linreg
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
for i in range(len(model_all.w)): 
    print("All model weight w%a: %a"% (i, model_all.w[i][0]))

def rmse(t, tp):
    error = t - tp
    square = error ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)       
    return root

predict_alpha = model_all.predict(X_test)
print("RMSE of all: %a"% rmse(t_test, predict_alpha))

plt.subplot()
plt.title("2D scatter plot - ALL using matrix A")
plt.plot([10,20,30,40,50],[10,20,30,40,50])
plt.scatter(model_all.predict(X_test), t_test, color='cyan', label="All")
plt.ylabel('Actual prices')
plt.xlabel('Predictions')
plt.legend(loc=2)
#plt.xlim(-18,45)
plt.savefig("2DscatterPlotModelALL.png")

