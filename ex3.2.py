import numpy as np


def Hebb(T):
    q = T.shape[0]
    r = T.shape[1]
    W = 0
    for k in range(r):
        W += np.dot(T[:, k].reshape(q, 1), T[:, k].reshape(1, q)) - np.eye(q)
    return W


T1 = np.array([[1, 1, 1, 1, 1]])
T2 = np.array([[1, -1, -1, 1, -1]])
T3 = np.array([[-1, 1, -1, -1, -1]])

T = np.column_stack((T1.T, T2.T, T3.T))
print T

print(Hebb(T))
