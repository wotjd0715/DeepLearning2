#오차역전파

#곱셈계층

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backforwrd(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

apple = 100
apple_n = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple,apple_n)
total_price = mul_tax_layer.forward(apple_price,tax)

print(total_price) #220.00000000000003


dprice = 1
apple_price_back, tax_back = mul_tax_layer.backforwrd(dprice)
apple_back, apple_n_back = mul_apple_layer.backforwrd(apple_price_back)
print(tax_back, apple_back, apple_n_back) #200 2.2 110.00000000000001 (미분값들)

# 덧셈계층
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x,y):
        self.x = x
        self.y = y
        return x+y

    def backforward(self,dout):
        dx = dout
        dy = dout
        return dx, dy

# ReLU 계층

class relu:
    def __init__(self):
        self.x= None
        self.y = None

    def forward(self,x):
        self.x = x
        if x>=0 :
            return x
        else:
            return 0

    def backforward(self,dout):
        if self.x >= 0:
            dx = dout
        else:
            dx = 0
        return dx

# ReLU 배열에서도 사용되게
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backforward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

import numpy as np
x = np.array([[1.0,-0.5], [-2.0,3.0]])
print(x)
ReLU = ReLU()
forward = ReLU.forward(x)
print(forward)
"""출력
[[ 1.  -0.5]
 [-2.   3. ]]
[[1. 0.]
 [0. 3.]]
"""
"""
dout 는 후의 loss함수 값을 의미 
신경망을 통과후 나온 값을 입력으로 넣고 ReLU함수를 지나 loss를 구할때 각 신경망의 입력에 따른 loss의 변화룰 구할때
ReLU의 backpropagation을 쓴다.
"""

# sigmoid 계층
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/ (1 + np.exp(-x))
        self.out = out

        return out
    def backforward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return  dx

# Softmax with Loss
class SoftmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backforward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


#오차역전파 방법을 적용한 신경망 구현하기
# epoch, learning_rate, hidden layer, construnction과 학습효율간의 관계 알기

import sys,os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import *

class TwoLayerNet_Backpropagation:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std = 0.01, batch_size=100):
        self.parms = {}
        self.parms['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.parms['b1'] = np.zeros(hidden_size1)
        self.parms['W2'] = weight_init_std * np.random.randn(hidden_size1,hidden_size2)
        self.parms['b2'] = np.zeros(hidden_size2)
        self.parms['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.parms['b3'] = np.zeros(output_size)
        self.parms['X1'] = np.zeros((batch_size,input_size))
        self.layers = {}



    def predict(self, x):
        self.parms['X1'] = x
        self.layers['Affine1'] = np.dot(self.parms['X1'], self.parms['W1']) + self.parms['b1']  # 100x784 784x200 = 100x200
        self.layers['Relu1'] = Relu1.forward(self, self.layers['Affine1'])  # 100x50
        self.layers['Affine2'] = np.dot(self.layers['Relu1'], self.parms['W2']) + self.parms['b2']  # 100x200 200x100 = 100x100
        self.layers['Relu2'] = Relu2.forward(self, self.layers['Affine2'])  # 100x100
        self.layers['Affine3'] = np.dot(self.layers['Relu2'], self.parms['W3']) + self.parms['b3']  # 100x100 100x10 = 100x10
        return self.layers['Affine3']

    def loss(self, x, t):
        y = softmax(self.predict(x))

        if y.ndim == 1:  # y가 1차원이라면
            t = t.reshape(1, t.size)  # 학습데이터 1개 가 t개
            y = y.reshape(1, y.size)

        loss = -np.sum(t*np.log(y + 1e-7), axis = 1)
        return loss
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self,x,t):
        self.loss(x,t)
        batch_size = t.shape[0] #100x10
        y = softmax(self.predict(x))

        dx = (y - t ) / batch_size
        grads = {}
        grads['W3'] = np.dot(self.layers['Relu2'].T, dx)
        grads['b3'] = np.sum(dx, axis=0)
        dx = np.dot(dx, self.parms['W3'].T)
        dx = Relu2.backward(self, dx)
        grads['W2'] = np.dot(self.layers['Relu1'].T,dx)
        grads['b2'] = np.sum(dx, axis = 0)
        dx = np.dot(dx, self.parms['W2'].T)
        dx = Relu1.backward(self,dx)
        grads['W1'] = np.dot(self.parms['X1'].T,dx)
        grads['b1'] = np.sum(dx, axis=0)

        return grads
    def learning(self,grad_backprop,learning_rate):
        self.parms['W1'] = self.parms['W1'] - learning_rate * grad_backprop['W1']
        self.parms['W2'] = self.parms['W2'] - learning_rate * grad_backprop['W2']
        self.parms['b1'] = self.parms['b1'] - learning_rate * grad_backprop['b1']
        self.parms['b2'] = self.parms['b2'] - learning_rate * grad_backprop['b2']
        self.parms['W3'] = self.parms['W3'] - learning_rate * grad_backprop['W3']
        self.parms['b3'] = self.parms['b3'] - learning_rate * grad_backprop['b3']
        return self.parms
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize= True, one_hot_label = True)
network = TwoLayerNet_Backpropagation(input_size=784, hidden_size1=100,hidden_size2=100, output_size=10)
train_size = x_train.shape[0]
batch_size = 100
epoch = 5000

learning_rate = 0.1

for i in range(epoch):
    batch_mask = np.random.choice(train_size,batch_size) #0~train_size까지중 batch_size만큼 무작위로 뽑아라
    x_batch = x_train[batch_mask] #학습데이터중 무작위 인덱스값 10개로 x_batch를 구성
    t_batch = t_train[batch_mask]
    loss = network.loss(x_batch,t_batch)
    accuracy =  network.accuracy(x_batch,t_batch)
    grad_backprop = network.gradient(x_batch,t_batch)
    network.learning(grad_backprop,learning_rate)
    if i % 500 == 0:
        print("epoch:" , "%04d" %(i + 1) , 'loss' , "%.4f" %(np.sum(loss)/100) , 'accuracy' , "%.4f" % (accuracy))

batch_mask = np.random.choice(x_test.shape[0],100) #0~train_size까지중 batch_size만큼 무작위로 뽑아라
print(batch_mask)
x_batch = x_test[batch_mask] #학습데이터중 무작위 인덱스값 10개로 x_batch를 구성
t_batch = t_test[batch_mask]
pred = np.argmax(softmax(network.predict(x_batch)), axis=1)
ansever = np.argmax(t_batch, axis =1)

print(pred)
print(ansever)

accuracy =  network.accuracy(x_test,t_test)
print(x_test.shape)
print(t_test.shape)
print(accuracy)
