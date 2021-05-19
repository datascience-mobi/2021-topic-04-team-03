import numpy
import numpy as np
import matplotlib.pyplot as plt



N=40000
mu, sigma = 210, 5
mu2, sigma2 = 10, 5
#X1 = np.random.normal(mu, sigma, N)
#X2 = np.random.normal(mu2, sigma2, N)
#X = np.concatenate([X1, X2])

randomNums1 = np.random.normal(25,20,5000)
randomInts1 = np.round(randomNums1)

randomNums2 = np.random.normal(200,30,4000)
randomInts2 = np.round(randomNums2)

X = np.concatenate([randomInts1, randomInts2])


plt.hist(X, np.arange(-0.5,256.5,1), color = 'steelblue')

plt.xticks(numpy.arange(0,256,15), fontsize = 7)
plt.title('Gray-value histogram')
plt.xlabel('Intensity', fontsize = 9)
plt.ylabel('Frequency', fontsize = 9)


plt.xlim(0,256)
plt.savefig('bimodal.pdf')