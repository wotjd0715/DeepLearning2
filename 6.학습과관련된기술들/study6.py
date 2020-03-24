# SGD구현하기
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self,params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate*grads[key]

class Momentum:
    def __init__(self, lr = 0.01, momentum= 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] = self.momentum* self.v[key] - self.lr*grads[key]
                params[key] += self.v[key]

# AdaGrad 구현하기
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key,val in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# 은닉층의 활성화값 분포
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size =  5
activations = {}
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i]=z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30 , range(0,2))
plt.show()

# Dropout 구현하기
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self,x,train_flg= True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x* self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self,dout):
        return dout* self.mask
