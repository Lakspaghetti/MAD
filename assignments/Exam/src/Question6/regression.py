from hashlib import new
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def loaddata(filename):
    """Load the accent data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',') #2D
    #print(Xt.shape)
    # Split into data matrix and target vector
    X = Xt[:,1:]
    t = Xt[:,0] #add : after 1 to get 2d array and also to read from left to right instead of top to bottom
    
    return t, X

# Load data
train_data_t, train_data_X = loaddata('accent-mfcc-data_shuffled_train.txt')
valid_data_t, valid_data_X = loaddata('accent-mfcc-data_shuffled_validation.txt')

# ADD YOUR SOLUTION CODE HERE!
def __rdmForests(trainingFeatures2D, trainingLabels):
    predictor = RandomForestClassifier(n_estimators = 100)
    predictor.fit(trainingFeatures2D, trainingLabels)
    return predictor

#Task a)
train_data_predictor = __rdmForests(train_data_X, train_data_t) #+1 to avoid 0
tdp_acc_score = train_data_predictor.score(train_data_X,train_data_t)
print("Task a)\n accuracy on training data", tdp_acc_score, "\n")


#look into the following page for further elaboration on methods and the class itself https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
def __optimalRMDF(samples, targets): 
    #define parameters first
    criterion = ["entropy", "gini"]
    maxTreeDepth = [2, 5, 7, 10, 15]
    nFeatures = ["sqrt", "log2"]
    

    Highest_Predictions = 0 #do not use train_data_predictor's score since it has 100% due to it being a training RDMF-classifier
    Highest_Avg_Prob = 0 #Same as above

    Most_optimal = train_data_predictor #train_data_predictor is only temporary
    optimal_parameters = ["",0,""] #temp values
    #3 nested for loops to perfrom exhaustive search. Total iterations is 20
    for c in criterion:
        for m in maxTreeDepth:
            for f in nFeatures:
                
                RDMF = RandomForestClassifier(criterion=c, max_depth=m, max_features=f)
                RDMF.fit(train_data_X, train_data_t)
                metric1 = (RDMF.predict(samples) == valid_data_t).sum()
                metric2 = np.mean([RDMF.predict_proba(samples)[i, int(valid_data_t[i])] for i in range(len(valid_data_t))]) #77 iterations per iteration in the nested forloops

                if Highest_Predictions == metric1 and Highest_Avg_Prob < metric2:
                    Highest_Predictions, Highest_Avg_Prob = metric1, metric2
                    print("") #create space
                    Most_optimal = RDMF
                    print("RDMF updated to: ")
                    optimal_parameters[0] = c
                    print("New criterion = ", c)
                    optimal_parameters[1] = m
                    print("New max tree depth = ", m)
                    optimal_parameters[2] = f
                    print("New number of features = ", f)
                    print("accuracy on valid data = ", RDMF.score(samples, targets))
                    print("number of correctly classified validation samples =", metric1)
                elif Highest_Predictions < metric1:
                    Highest_Predictions, Highest_Avg_Prob = metric1, metric2
                    print("") #create space
                    Most_optimal = RDMF
                    print("RDMF updated to: ")
                    optimal_parameters[0] = c
                    print("New criterion = ", c)
                    optimal_parameters[1] = m
                    print("New max tree depth = ", m)
                    optimal_parameters[2] = f
                    print("New number of features = ", f)
                    print("accuracy on valid data = ", RDMF.score(samples, targets))
                    print("number of correctly classified validation samples =", metric1)
    return Most_optimal, optimal_parameters     

print("task b and c)")
ORDMF, params = __optimalRMDF(valid_data_X, valid_data_t)



