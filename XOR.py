# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def hardlim(a):  # 阈值函数
    a[a >= 0] = 1
    a[a < 0] = 0
    return a


max_epoch = 20  # 最大循环次数

p1 = np.array([(0, 0, 1, 1), (0, 1, 0, 1)])
t1 = np.array([(0, 1, 0, 0), (0, 0, 1, 0)])
w1 = np.random.randn(2, 2)
b1 = np.random.randn(2, 1)

p2 = t1
t2 = np.array([(0, 1, 1, 0)])
w2 = np.random.randn(2, 1).T
b2 = np.random.randn(1, 1)

a1 = hardlim(np.dot(w1, p1) + b1)
for epoch1 in range(1, max_epoch):
    if (a1 == t1).all():
        epoch1 -= 1
        break
    e1 = t1 - a1
    dw1 = np.dot(e1, p1.T)
    db1 = np.dot(e1, np.ones((4, 1)))
    w1 += dw1
    b1 += db1
    a1 = hardlim(np.dot(w1, p1) + b1)
print 'epoch1 = ', epoch1

a2 = hardlim(np.dot(w2, p2) + b2)
for epoch in range(1, max_epoch):
    if (a2 == t2).all():
        epoch -= 1
        break
    e2 = t2 - a2
    dw2 = np.dot(e2, p2.T)
    db2 = np.dot(e2, np.ones((4, 1)))
    w2 += dw2
    b2 += db2
    a2 = hardlim(np.dot(w2, p2) + b2)
print 'epoch2 = ', epoch

print 'w1 = ', w1
print 'b1 = ', b1
print 'w2 = ', w2
print 'b2 = ', b2
#huihdsiufjdshfhjdsf


