# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as axes
from mpl_toolkits.mplot3d import Axes3D


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


# rdm1 = np.random.RandomState(1)
# rdm2 = np.random.RandomState(2)
#
# e = rdm1.random_integers(-2, 2, 20).reshape(1, 20)
# ec = rdm2.random_integers(-2, 2, 20).reshape(1, 20)
# p = np.zeros((2, 20))
#

# 初始化输入变量
e, ec = np.zeros((2, 25))
p = np.zeros((2, 25))
n1 = int(-3)
n2 = int(-2)
for i in range(25):
    if i % 5 == 0:
        n1 += 1
        n2 = -2
    e[i] = n1
    ec[i] = n2
    n2 += 1

p[0] = e
p[1] = ec

t = (e + ec) / 2
t = t.astype('int')

print '输入p = '
print p
print t

s1 = 10

# w1 = 1 - 2 * np.random.random_sample((s1, 2))
# b1 = 1 - 2 * np.random.random_sample((s1, 1))
# w2 = 1 - 2 * np.random.random_sample((1, s1))
# b2 = 1 - 2 * np.random.random_sample()

rdm3 = np.random.RandomState(10)
w1 = 1 - 2 * rdm3.random_sample((s1, 2))
b1 = 1 - 2 * rdm3.random_sample((s1, 1))
w2 = 1 - 2 * rdm3.random_sample((1, s1))
b2 = 1 - 2 * rdm3.random_sample()
# print 'w1 = ', w1
# print 'b1 = ', b1
# print 'w2 = ', w2
# print 'b2 = ', b2

a1 = tansig(np.dot(w1, p) + b1)
a2 = np.dot(w2, a1) + b2

e = t - a2
SSETemp = SSE(e)
max_epoch = 100000
eta = 0.017
error_goal = 0.001

dw11 = np.ones((s1, 2))
db11 = np.ones((s1, 1))
dw22 = np.ones((1, s1))
db22 = np.ones((1, 1))

for epoch in range(max_epoch):
    if (SSE(e) < error_goal):
        break
    # 自适应学习速率eta
    if (SSE(e) < SSETemp):
        eta = 1.05 * eta
    elif (SSE(e) > SSETemp):
        eta = 0.7 * eta
    else:
        eta = eta
    SSETemp = SSE(e)
    #################
    dw2 = eta * np.dot(e, a1.T)
    db2 = eta * e.sum()
    w2 += dw2
    b2 += db2

    df1 = dtansig(np.dot(w1, p) + b1)
    e0 = np.dot(w2.T, e)
    delta = e0 * df1
    dw1 = eta * np.dot(delta, p.T)
    db1 = eta * delta

    w1 += dw1
    b1 += catb(db1)

    a1 = tansig(np.dot(w1, p) + b1)
    a2 = np.dot(w2, a1) + b2
    e = t - a2

    if epoch == 1:
        print dw1

print epoch
print 'SSE(e) = ', SSE(e)
print dw2
# print 'w1 = ', w1
# print 'b1 = ', b1
# print 'w2 = ', w2
# print 'b2 = ', b2

# at1 = tansig(np.dot(w1, [[1], [-1]]) + b1)
# at2 = np.dot(w2, at1) + b2
# print 'at2 = ', at2

# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# print X
# X, Y = np.meshgrid(X, Y)
# print X
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# rainbowfig = plt.figure('origin')
# ax = Axes3D(fig)
# e, ec = np.meshgrid(e, ec)
# ax.plot_surface(e, ec, t, rstride=1, cstride=1)
# fig = plt.figure('model')
# ax = Axes3D(fig)
# ax.plot_surface(e, ec, a2, rstride=1, cstride=1)
# plt.show()

# ax = plt.subplot(111, projection='3d')
# ax.scatter(e[0],ec[0],t[0])
# ax.set_zlabel('t')
# ax.set_ylabel('ec')
# ax.set_xlabel('e')
# plt.show()
