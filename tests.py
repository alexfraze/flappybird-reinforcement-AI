import numpy as np

x =np.array([1,2])
print(x)
print(np.delete(x, 0))
# print(np.vstack((x, np.array([3,4,5]))))
x = np.array([np.array([1,2,3]),np.array([1,2,3])])
print(x)
y = list(x)
print(y)
del y[0]
print(y)
print(np.array(y))
