
"""
#활성화 함수인 계다 함수 구현하기
import numpy as np

def step_function(x):
    if x>0:
         return 1
    else:
        return 0

def step_function_for_array(x):
    y = x>0
    return y.astype(np.int)

"""
x = [0.1, -1, 2]
y = [False, False, True]
"""

import matplotlib.pylab as plt

def step_function2(x):
    return np.array(x>0 , dtype= np.int)

x = np.arange(-5.0, 5.0 , .1)
y = step_function2(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1) # y축의 범위 지정
plt.show()

#시그모이드 함수 구현하기

def sigmoid(x):
    return 1/(1+ np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y2 = sigmoid(x)
plt.plot(x,y2)
plt.ylim(-0.1,1.1)
plt.show()

#step과 sigmoid동시에 출력
plt.plot(x,y, label = "step_function")
plt.plot(x,y2, linestyle = '--', label = "sigmoid")
plt.legend()
plt.show()

#ReLU함수 구현하기

def relu(x):
    return np.maximun(0,x)
"""
#다차원 배열의 계산
import numpy as np

A = np.array([1,2,3,4,5])
print(A)

print( np.ndim(A) )

print( A.shape )

print( A.shape[0] )

"""출력
[1 2 3 4 5]

1

(5,)

5
"""

B = np.array([[1,2],[3,4],[5,6]])
print(B)

print( np.ndim(B) )

print( B.shape )

print( B.shape[0] )

"""출력
[[1 2]
 [3 4]
 [5 6]]
 
2

(3, 2)

3
"""

#행렬 곱

A = np.array([[1,2],[3,4]])
print(A.shape)
B= np.array([[5,6],[7,8]])
print(B.shape)

print( np.dot(A,B) )

print( np.matmul(A,B))
"""출력
(2, 2)
(2, 2)

[[19 22]
 [43 50]]

[[19 22]
 [43 50]]
"""

#주의 array([1,2,3,...,n]) 은 nx1행렬 1xn행력아님

k = np.array([[1],[2],[3],[4]])
print(k.shape)

i = np.array([1,2,3,4])
print(i.shape)

o = np.array([[1,2,3,4]])
print(o.shape)
"""출력
(4, 1)
(4,)
(1, 5)
"""

#신경망에서의 행렬곱

X = np.array([1,2])
print(X.shape)

W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape)

Y = np.dot(X,W)
print(Y)

# 3층 신경망 구현하기

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5] ,[0.2,0.4,0.6]]) #2x3
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5],[0.3,0.6]]) #3x2
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]]) #2x2
    network['b3'] = np.array([0.1,0.2])

    return network

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    z1 = sigmoid(np.dot(x,W1)+b1)
    print("z1\n")
    print(z1)
    z2 = sigmoid(np.dot(z1,W2)+b2)
    print("z2\n")
    print(z2)
    z3 = sigmoid(np.dot(z2,W3)+b3)
    print("z3\n")
    print(z3)
    y = np.argmax(z3) #가장큰 인덱스의 번호 출력
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)
"""출력
[0.57855079 0.66736228]
1
"""

#softmax함수 만들기
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)#오버플로 방지를 위해 c를 뺴준다
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(np.sum(y))
"""출력
[0.01821127 0.24519181 0.73659691]
1.0
"""

# 손글씨 숫자 인식

# 이번 과정에선 학습 과정은 생략하고 추론 과정만 구현해보자

import sys, os
sys.path.append(os.pardir) # 부모 디렉토리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist # dataset/minist의 load_mnist함수를 import

(x_train, t_train),(x_test, t_test) = \
    load_mnist(flatten = True, normalize = False) #load_minst함수로 읽은 MNIST데이터를 "(훈련 이미지, 훈련 레이블),(시험이미지, 시험레이블)" 형식으로 반환
# normalize는 입력이미지의 픽셀값을 0.0 ~ 1.0 사이의 값으로 정규화 할지 정합니다. False로 설정하면 입력 이미지의 픽셀은 원래 값 그대로 0~255사이의 값을 유지합니다.
# flatten은 입력 이미지를 1차원 배열로 만들지 정합니다. False로 할경우 1x28x28의 3차원 배열로
# one_hot_label은 레이블을 원-핫 인코딩 형대로 정합니다.
"""
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, )
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000, )

"""

#다운받은 이미지 한장을 확인해보자
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label) # 5 / 출력할 이미지에 저장된 라벨값을 출력

print(img.shape) #(784, ) / flatten = True로 1차원 배열로 만들어둔 이미지를 다시 28x28형태로 만들어 줘야 한다.
img = img.reshape(28,28)

img_show(img)

# 신경망의 추론 처리
"""
입력층 뉴런은 28x28이미지를 1차원 배열로 만든 784개, 출력층 뉴런은 0~9까지 분류하기위해 10개로 만듬
"""
import pickle
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label = False)
    return x_test, t_test
"""
load_mnist 함수의 인수인 normalize를 True로 설정했음. True로 설정시 0~255범위인 각 픽셀값을 0.0 ~ 1.0 범위로 변환(단순히 픽셀의 값을 255로 나눔)
이처럼 데이터를 특정 범위로 변환하는 처리를 '정규화'라 하고, 신경망의 입력 데이터에 특정 변환을 가하는 것을 '전처리'라 함
"""


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network #picle파일인 sample_weight.pkl에 저장돤 학습된 가중치 매개변수를 읽음


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#데이터 분류의 정확성을 측정해 보자
x, t = get_data() #MNIST 데이터 셋을 얻고 네트워크를 형성
network = init_network()
accuracy_cnt = 0

for i in range(len(x)): # for문을 돌리며 x에 저장된 이미지 데이터를 1개씩 꺼내 predict()함수로 분류합니다.
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:   #예측한 값과 레이블의 값을 비교하여 맞을경우 count에 +1을 해줍니다.
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 배치처리

x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape) #이미지 하나의 shape
print(W1.shape)
print(W2.shape)
print(W3.shape)

print(y.shape) # 0~9까지 분류
print(y[0].shape) # 마지막으로 들어온 이미지가 0일 확률을 가지고있는(즉, 실수의 shape)
print(y[0]) # 마지막으로 들어온 이미지가 0일 확률
print(np.argmax(y)) # 마지막으로 들어온 이미지의 예측값
"""
출력
(10000, 784) # 입력 이미지 수가 10000개
(784,)
(784, 50)
(50, 100)
(100, 10)

(10,)
()
0.0004288287
6
"""

# x,y 같이 하나로 묶은 입력 데이터를 배치(batch)라고 함

"""
x, t = get_data() #MNIST 데이터 셋을 얻고 네트워크를 형성
network = init_network()
accuracy_cnt = 0

for i in range(len(x)): # for문을 돌리며 x에 저장된 이미지 데이터를 1개씩 꺼내 predict()함수로 분류합니다.
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:   #예측한 값과 레이블의 값을 비교하여 맞을경우 count에 +1을 해줍니다.
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
"""

#정확도를 예측했던 코드를 배치를 이용해 구현해보자

x, t = get_data()
network = init_network()
batch_size = 100 #배치 크기
accuracy_cnt = 0

for i in range(0,len(x),batch_size): # i를 0 부터 x크기-1 까지 batch_size만큼 늘림 즉 i는 0, 100 , 200 ...
    x_batch = x[i : i+batch_size]
    y_batch = predict(network, x_batch)
    p= 
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
np.argmax(y_batch, axis = 1)
# axis = 1 설명
x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3]]) # 2x3행렬
y = np.argmax(x, axis = 0)
print(y)
y = np.argmax(x, axis = 1)
print(y)
y = np.argmax(x, axis = 2)
print(y)

"""출력
[1 0 1] #[0.1, 0.3, 0.2], [0.8,0.1,0.5], [0.1,0.6,0.3] 중 가장 큰 인덱스값 0차원
[1 2 1] #[0.1,0.8,0.1], [0.3,0.1,0.6], [0.2,0.5,0.3]중 가장 큰 인덱스 값 1차원
axis 2 is out of bounds for array of dimension 2
"""