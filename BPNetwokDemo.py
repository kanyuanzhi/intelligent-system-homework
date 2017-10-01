# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def tansig(x):  # 双曲正切S型函数
    result = (1 - np.exp(-x)) / (1 + np.exp(-x))
    return result


def dtansig(x):  # 双曲正切S型函数的导数
    result = 2 / (np.exp(x) + np.exp(-x) + 2)
    return result


def catb(b):
    new_b = np.ones((b.shape[1], 1))
    return np.dot(b, new_b)


def SSE(a):
    return (a ** 2).sum() / 2


p = np.arange(-1, 1.1, 0.1).reshape(1, 21)
t = np.array([[-0.96, -0.577, -0.0729, 0.377, 0.641, 0.66, 0.461, 0.1336, -0.201, -0.434, -0.5,
               -0.393, -0.1647, 0.0988, 0.3072, 0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]])

# w1 = np.random.randn(5, 1)
# b1 = np.random.randn(5, 1)
# w2 = np.random.randn(1, 5)
# b2 = np.random.randn()

s1 = 5

rdm = np.random.RandomState(10)
w1 = (1 - 2 * rdm.random_sample((s1, 1))) * 10  # 权值的量级为s1根号r（s为隐含层神经元数，r为输入节点数）
b1 = 1 - 2 * rdm.random_sample((s1, 1))
w2 = (1 - 2 * rdm.random_sample((1, s1))) * 10
b2 = 1 - 2 * rdm.random_sample()

# random_samplew1 = 1 - 2 * np.random.random_sample((5, 1))
# b1 = 1 - 2 * np.random.random_sample((5, 1))
# w2 = 1 - 2 * np.random.random_sample((1, 5))
# b2 = 1 - 2 * np.random.random_sample()

a1 = tansig(np.dot(w1, p) + b1)
a2 = np.dot(w2, a1) + b2
e = t - a2
print e

eta = 0.016
max_epoch = 100000
error_goal = 0.01
plt.figure('target & process')

for i in range(max_epoch):
    if (SSE(e) < error_goal):
        break
    # if (SSE(e) > 20):
    #     break
    # if (i % 5000 == 0):
    #     plt.plot(p[0], a2[0])
    dw2 = eta * np.dot(e, a1.T)
    db2 = eta * e.sum()
    w2 += dw2
    b2 += db2

    df1 = dtansig(np.dot(w1, p) + b1)  # 5*21
    e0 = np.dot(w2.T, e)  # 5*21
    delta = e0 * df1
    dw1 = eta * np.dot(delta, p.T)
    db1 = eta * delta
    w1 += dw1
    b1 += catb(db1)

    a1 = tansig(np.dot(w1, p) + b1)
    a2 = np.dot(w2, a1) + b2
    e = t - a2

print SSE(e)
print i
print w1
print b1
print w2
print b2
print np.dot(w2, tansig(w1 * 0.15 + b1)) + b2

plt.plot(p[0], t[0], '.')
plt.plot(p[0], a2[0])

plt.show()
