import numpy as np
import linreg
import matplotlib.pyplot as plt

#load data
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

FPRunTimes, InputVar = raw[:, 1], raw[:, 0]

lam = np.logspace(-8, 0, 100, base=10)
#print(lam)

regularised_model = linreg.LinearRegression()

def GetBestLamb(x, t, k):
    loss = regularised_model.LOOCV(x, t, lam[0], k)
    for i in range(1,100):
        lossOfi = regularised_model.LOOCV(x, t, lam[i], k)
        if loss > lossOfi:
            loss = lossOfi
            result = lam[i]
            #print("lamInside = %.10f"% lam[i])
    return result
"""
print("lam = %a" % GetBestLamb(InputVar,FPRunTimes,1))
regularised_model.fit(InputVar, FPRunTimes, GetBestLamb(InputVar,FPRunTimes,1), 1)
print("w of best lam: %a"% regularised_model.w)    
regularised_model.fit(InputVar, FPRunTimes, lam[0], 1)
print("w of lam 0: %a"% regularised_model.w)  


LOOCVList = np.array([regularised_model.LOOCV(InputVar, FPRunTimes, i, 1) for i in lam])

plt.subplot()
plt.title("LOOCV first order")
plt.scatter(lam,LOOCVList , color='cyan', label="error")
plt.ylabel('LOOCV error')
plt.xlabel("$\lambda$")
plt.legend(loc=2)
plt.xscale("log")
plt.savefig("LOOCVofFirstOrder.png")
"""
#b

LOOCVList = np.array([regularised_model.LOOCV(InputVar, FPRunTimes, i, 4) for i in lam])

print("lam = %a" % GetBestLamb(InputVar,FPRunTimes,4))
regularised_model.fit(InputVar, FPRunTimes, GetBestLamb(InputVar,FPRunTimes,4), 4)
print("w of best lam: %a"% regularised_model.w)    
regularised_model.fit(InputVar, FPRunTimes, lam[0], 4)
print("w of lam 0: %a"% regularised_model.w)  

plt.subplot()
plt.title("LOOCV fourth order")
plt.scatter(lam,LOOCVList , color='cyan', label="error")
plt.ylabel('LOOCV error')
plt.xlabel("$\lambda$")
plt.legend(loc=2)
plt.xscale("log")
plt.savefig("LOOCVofFourthOrder.png")