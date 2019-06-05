import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets.samples_generator import make_blobs

n_components = 4
# X, truth = make_blobs(n_samples=400, centers=n_components, 
#                       cluster_std = 2 ,
#                       random_state=42)


X = np.array([]).reshape(0,2)
C = np.array([])
i = 1
for n in range(n_components):
    x = np.random.randn(100,2)+i*3
    X = np.append(X,x,axis=0)
    C = np.append(C,np.ones(100)*i)
    i=i+1

x = X[:, 0]
y = X[:, 1]
plt.scatter(X[:, 0], X[:, 1], s=50, c = C)
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("x") 
plt.ylabel("y");
plt.show()
# Extract x and y
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10

xmin = min(x) - deltaX
xmax = max(x) + deltaX

ymin = min(y) - deltaY
ymax = max(y) + deltaY

print(xmin, xmax, ymin, ymax)

# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)


fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plt.show()