import numpy as np

points = np.loadtxt('points.txt', delimiter=',')

C = np.cov(points)

PCevals, PCevecs = np.linalg.eig(C) 

print("this is C:\n", C)
print("this is cevals:\n", PCevals)
print("this is cevecs:\n", PCevecs)
