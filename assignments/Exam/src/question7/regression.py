import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def loaddata(filename):
    # Load data
    X = np.loadtxt(filename, delimiter=',')
    
    return X

# Load data
X = loaddata('seedsDataset.txt')
X = (X - X.mean(0))/X.std(0)


def KMC(K, data, maxI):
    idx = np.random.randint(data.shape[0]-1, size=K)
    oldCentroids = np.empty(shape=(K, 8))
    centroids = data[idx,:]
    i=0#number of iterations
    ClusterList = []

    while i<maxI and not np.array_equal(oldCentroids, centroids):
        #print("iteration: ", i)
        idxclusters = np.full(shape=(K,200), fill_value=data.shape[0], dtype=np.int64)

        for j in range(data.shape[0]): #create idxlist of min dist to each centroid
            idist = np.empty(K) #the distances from i to all centroids
            for k in range(K):
                idist[k] = np.sum((data[j] - centroids[k])**2)

            idxclusters[np.argmin(idist), j] = j #the idx of each element put into a cluster list
            
        for l in range(K): #compute number of samples in each cluster and update centroids
            #print("cluster_%a contains"% l,":",((idxclusters[l])[idxclusters[l] != data.shape[0]]).shape[0], "samples")
            ClusterK = data[(idxclusters[l])[idxclusters[l] != data.shape[0]],:]
            oldCentroids[l] = centroids[l] #update old centroids before the new ones
            centroids[l] = (np.sum(ClusterK, 0))/(idxclusters[l])[idxclusters[l] != data.shape[0]].shape[0]#update centroids
        
        for m in range(K):
            ClusterList.append(data[(idxclusters[m])[idxclusters[m] != data.shape[0]],:])

        i += 1

    return centroids, idxclusters, ClusterList


###INTRA CLUSTER AND N SAMPLES FROM EACH CLUSTER IN FINAL SOLUTION
def IntraCluster(centroids, Clusters):
    distance = 0

    for i in range(centroids.shape[0]):
        distance = distance + np.sqrt(np.sum((Clusters[i]-centroids[i])**2))

    return distance


Cent, IDXClus, Clus = KMC(3, X, 1000)
ShortestIC = IntraCluster(Cent, Clus)

for run in range(4): #amount of runs is set to 4 since first run is made outside of the loop
    tempCent, tempIDXClus, tempClus = KMC(3, X, 1000)
    tempIC = IntraCluster(tempCent, tempClus)
    if tempIC < ShortestIC:
        Cent, IDXClus, Clus = tempCent, tempIDXClus, tempClus
        ShortestIC = tempIC

###COMPUTE NUMBER OF SAMPLES AND PRINT
for n in range(Cent.shape[0]):
    print("cluster_%a contains"% n,":",((IDXClus[n])[IDXClus[n] != X.shape[0]]).shape[0], "samples")

###PCA FROM HERE
SKLPCA = PCA(n_components=2)

TwoDimX = SKLPCA.fit_transform(X)
TwoDimCent = SKLPCA.fit_transform(Cent)

Clus1 = TwoDimX[(IDXClus[0])[IDXClus[0] != 200],:]
Clus2 = TwoDimX[(IDXClus[1])[IDXClus[1] != 200],:]
Clus3 = TwoDimX[(IDXClus[2])[IDXClus[2] != 200],:]

###plotting from here
plt.subplot()
plt.title("Scatterplot")
plt.scatter(TwoDimX[:,0], TwoDimX[:,1], color='grey')
plt.scatter(Clus1[:,0], Clus1[:,1], color='blue')
plt.scatter(Clus2[:,0], Clus2[:,1], color='green')
plt.scatter(Clus3[:,0], Clus3[:,1], color='red')
plt.scatter(TwoDimCent[:,0], TwoDimCent[:,1], color='black', s=100) 
plt.savefig("test.png")