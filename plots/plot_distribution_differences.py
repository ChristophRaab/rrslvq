import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches

import matplotlib.lines as mlines
w_size = 30
p = 0.001
mean = 0

X = np.random.normal(size=(w_size,2))

alpha = [1,1]

while all(a > p  for a in alpha):
    mean = mean + 0.01
    Y = np.random.normal(loc=mean,size=(w_size,2)) 
    #Y = np.random.binomial(1,0.5,(w_size,2)) 
    alpha = [stats.ks_2samp(x,y).pvalue for x,y in zip(X.T,Y.T)]

print("Difference in mean:"+str(abs(mean-0)))


fig, ax = plt.subplots()
red, = plt.plot(X[:,0],X[:,1],'ro')
mean = np.mean(X,axis=0)
black, = plt.plot(mean[0],mean[1],"bx")
mean = np.mean(Y,axis=0)
yellow, = plt.plot(mean[0],mean[1],"yx")
green, = plt.plot(Y[:,0],Y[:,1],'go')
plt.legend([red, green,black,yellow], ["Old Concept", "New Concept","Mean Old","Mean New"])
ax.set_title('Concept Comparison')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

plt.show()

