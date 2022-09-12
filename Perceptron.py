'''
输入：训练集T={(x1,y1),…,(xN,yN)}，学习率eps
输出：w, b，感知机模型f(x) = sign(w * x + b)

(1)	选取初值w0, b0
(2)	在训练集中选取数据（xi, yi）
(3)	如果yi * (w*xi + b) <= 0:
w = w + eps * yi * xi
b = b + eps * yi
(4)转到第二步，直至训练集中没有误分类点

'''

import numpy as np
import matplotlib.pyplot as plt
def Perceptron(w, b, x, y, eps):
    checkOK = True
    while checkOK:
        for i in range(0, x.shape[1]):
            temp = w * x[:, i]
            while y[i] * (temp + b) <= 0:
                w = w + np.dot(eps * y[i], x[:, i].T)
                b = b + eps * y[i]
                temp = w * x[:, i]
                
        ifBreak = False
    
        for j in range(0, x.shape[1]):
            temp = w * x[:, j]
            if y[j] * (temp + b) > 0:
                continue
            else:
                ifBreak = True
                break
                
        if ifBreak:
            continue
        else:
            checkOK = False


    print(w, b)

    zp = []
    zn = []
    for i in range(0, x.shape[1]):
        if y[i] == 1:
            for j in range(0, len(x)):
                zp.append(x[j, i])
        else:
            for j in range(0, len(x)):
                zn.append(x[j, i])
    plt.scatter(zp[0: len(zp): len(x)], zp[1: len(zp): len(x)])
    plt.scatter(zn[0: len(zn): len(x)], zn[1: len(zn): len(x)])

    x1 = np.arange(0, 5, 0.1)
    x2 = np.array(-(w[0, 0] * x1 + b)/w[0, 1]).squeeze()
    plt.plot(x1, x2)
    plt.show()

def runPerceptron():
    x1 = np.matrix([[3, 3], [4, 3], [1, 1]])
    x = x1.T
    y1 = np.matrix([1, 1, -1])
    y = y1.T
    w = np.matrix([0, 0])
    b = 0
    Perceptron(w, b, x, y, 1)