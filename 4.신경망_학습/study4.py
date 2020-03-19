import numpy as np
#오차제곱합
y1 = [0.2,0.3,0.5] #예측값을 softmax로 표현
y2 = [0.5,0.3,0.2]
y3 = [0,0,1]
t = [0,0,1] #정답레이블을 원-핫 인코딩으로 표현

def sum_square_error(y,t):
    return 0.5 * np.sum( (y-t)**2 )

print( sum_square_error( np.array(y1), np.array(t)))
print( sum_square_error( np.array(y2), np.array(t)))
print( sum_square_error( np.array(y3), np.array(t)))

""" 출력
0.19  # 오차율
0.49000000000000005
0.0
"""

#교차 엔트로피
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t* np.log(y+delta)) #delta를 넣어주는 이유는 만약 y=0일때 무한대로 발산해버리기에 아주 작은값을 더해준다

y4 = [0.5,0.5,0]

print(cross_entropy_error(np.array(y4),np.array(t)))
"""출력
delta가 없을경우
  return -np.sum(t* np.log(y+delta))
inf

delta가 있을경우
16.11809565095832
"""
print(cross_entropy_error(np.array(y1),np.array(t)))
print(cross_entropy_error(np.array(y2),np.array(t)))
print(cross_entropy_error(np.array(y3),np.array(t)))
"""출력
0.6931469805599654
1.6094374124342252
-9.999999505838704e-08
"""

#미니배치학습

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize= True, one_hot_label = True)

print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, 10)
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000, 10)

train_size = x_train.shape[0]

print(x_train.shape[0]) # 60000
print(x_train.shape[1]) # 784

batch_size = 10
batch_mask = np.random.choice(train_size,batch_size) #0~train_size까지중 batch_size만큼 무작위로 뽑아라
x_batch = x_train[batch_mask] #학습데이터중 무작위 인덱스값 10개로 x_batch를 구성
t_batch = t_train[batch_mask]

# 교차엔트로피에 적용 (레이블이 원-핫 인코딩의 경우)
def cross_entropy_batch (y,t):
    if y.ndim == 1: #y가 1차원이라면
        t = t.reshape(1, t.size) #학습데이터 1개 가 t개
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] #학습데이터의 갯수
    return -np.sum(t*np.log(y + 1e-7))/ batch_size

# 교차엔트로피에 적용 (레이블이 실수인 경우)
def cross_entropy_batch2 (y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))
"""
np.arange(batch_size)는 0부터 batch_size - 1까지 배열을 생성 => y[0, t[0]], y[1,t[1]], ...
"""

#2층 신경망 클래스 구현하기

import sys, os
sys.path.append(os.pardir)
from common.common.functions import *
from common.common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {} #신경망의 매개변수들을 저장할 딕셔너리 변수 (class의 인스턴스 변수)
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = sigmoid(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t , axis =1 )

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {} # 매개변수들의 기울기를 보관할 딕셔너리 변수 (class의 인스턴스 변수)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

net = TwoLayerNet(input_size=784, hidden_size = 100, output_size= 10)

print(net.params['W1'].shape) #(784, 100)
print(net.params['b1'].shape) #(100,)
print(net.params['W2'].shape) #(100, 10)
print(net.params['b2'].shape) #(10,)
"""
x = np.random.rand(100,784)
t = np.random.rand(100, 10) # 10개의 랜덤값 원소를 가진 배열을 100개 생성
grads = net.numerical_gradient(x,t)


print(grads['W1'].shape) #(784, 100)
print(grads['b1'].shape) #(100,)
print(grads['W2'].shape) #(100, 10)
print(grads['b2'].shape) #(10,)

accu = net.accuracy(x,t)
print(accu)#0.13  레이블 t를 랜덤으로 주었기에 정확도가 매우 떨어진다
"""

# 미니배치 학습 구현하기

import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize= True, one_hot_label = True)

train_loss_list = []

epoch = 2000 #반복횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate =0.1

network = TwoLayerNet(input_size=784, hidden_size= 100, output_size= 10)

save_i = []
save_loss = []

for i in range(epoch):

    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch,t_batch)
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print('epoch:' + '%04d' % (i+1) + 'loss:' + '%.3f' % loss)
    save_i.append(i)
    save_loss.append(loss)

plt.plot(save_i, save_loss, label="loss")
plt.legend()
plt.show()


#시험 데이터로 평가하기

import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize= True, one_hot_label = True)

train_loss_list = []

epoch = 2000 #반복횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate =0.1

network = TwoLayerNet(input_size=784, hidden_size= 100, output_size= 10)

save_i = []
save_i_acc = []
save_loss = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size,1) # 1 epoch당 반복수

for i in range(epoch):

    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch,t_batch)
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print('epoch:' + '%04d' % (i+1) + 'loss:' + '%.3f' % loss)
    save_i.append(i)
    save_loss.append(loss)

    if i % iter_per_epoch == 0:
        save_i_acc.append(i)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plt.plot(save_i, save_loss, label="loss")
plt.plot(save_i_acc, train_acc_list, linestyle = "--", label = "train_accuracy")
plt.plot(save_i_acc, test_acc_list, linestyle = ":", label = "test_accuracy")
plt.legend()
plt.show()

# trian_accuracy선과 test_accuracy선이 거의 일치함을 통해 오버피팅이 거의 일어나지 않았음을 알 수 있음