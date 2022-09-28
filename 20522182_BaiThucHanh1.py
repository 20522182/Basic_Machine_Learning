import numpy as np
X= 2* np.random.rand(100,1)
X_b = np.c_[np.ones((100,1)),X]
    
import matplotlib.pyplot as plt
plt.plot(X,y,"b.")

X_b.shape

thera_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
X_new=np.array([[0],[2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = np.dot(X_new_b,thera_best)

plt.plot(X,y,"b.")
plt.plot(X_new_b, y_pred,"r-")