##파이썬 인터프리터

# 리스트 생성
"""
a = [1,2,3,4,5]

print(a[-1])
print(a[:-1])
print(a[:4])
"""
""" 출력
5
[1, 2, 3, 4]
[1, 2, 3, 4]
"""

# 딕셔너리 생성
me = {'height' : 100}
print(me['height'])
me['weight'] = 40
print(me)

""" 출력
100
{'height': 100, 'weight': 40}
"""

#for문과 list
for i in [1,4,5]:
    print(i)  #리스트 안의 값을 하나씩 출력 1,4,5

a = [1,4,5]
for i in range(len(a)):
    print(i) # 리스트의 길이(인덱스 값)을 출력 0,1,2

#함수
def hello():
    print("hello world")

hello() # hello world 출력

def hello(object):
    print("hello " + object)
hello("kitty") #hello kitty출력

##파이썬 스크립트

# $ python filename.py

#클래스를 만들어 개발자가 직접 정의하며 독자적인 자료형을 만들수 있다.
class classname:
    def __init__(self, var1, var2):
        print()
    def method1(self, var1):
        print()
    def method2(self, var1, var2):
        print()

class hello:
    def __init__(self, name):
        self.name = name
        print("initialized")

    def friend(self):
        print("hello" + self.name)

    def family(self, mom):
        print(mom)
        mom = "asaa"
        print(mom)
        self.mom = mom
        print(self.name + self.mom + mom) #self.mom이랑 mom이랑 뭔차이지?
        self.__dad = "asd" # __가 앞에만 붙으면 비공개 속성으로 class밖에서 사용할수 없음
h = hello("katar")
h.friend()
h.family("hook")
"""
출력
initialized
hellokatar
hook
asaa
katarasaaasaa
"""
print("--------------넘파이--------------------")
#넘파이 가져오기
import numpy as np

#넘파이 배열 생성하기
x = np.array([1.0,2.0,3.0])
print(x)
print(type(x))

"""
출력
[1. 2. 3.]
<class 'numpy.ndarray'>
"""

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x-y)
print(x * y)
print(x/y)
# 단 array의 원소수가 같아야 연산을 실행할수있다.

print(x/ 2.0)
#스칼라와의 연산은 각 원소별로 한 번씩 수행된다. 이 기능을 브로드캐스트라고 한다.
"""
출력 
[-1. -2. -3.]
[ 2.  8. 18.]
[0.5 0.5 0.5]

[0.5 1.  1.5]
"""

#넘파이의 n차원 배열

A =  np.array([ [1,2] , [3,4] ])
print(A)
print(A.shape)
print(A.dtype)
"""
출력
[[1 2]
 [3 4]]
 
(2, 2)

int32
"""

#행렬의 산술연산

B = np.array([ [3,0] , [0,6] ])
print(A+B)
print(A*B)
print(A * 10)
"""
출력
[[ 4  2]
 [ 3 10]]
 
[[ 3  0]
 [ 0 24]]
 
 [[10 20]
 [30 40]]
"""
#형상이 같은 행렬끼라면 행렬의 산술 연산도 대응하는 원소별로 계산
#수학에서는 1차원 배열 = 벡터 , 2차원 배열 = 행렬, 3차원 배열 = 다차원 배열
#벡ㅌ와 행렬을 일반화 한것이 Tensor

A = np.array([[1,2],[3,4]])
B = np.array([10,20])
print(A*B)
"""
출력
[[10 40]
 [30 80]]
"""
#원소 접근
X = np.array([[0,1],[2,3],[4,5]])
print(X)

print(X[0])

print(X[0][0])
"""
출력
[[0 1]
 [2 3]
 [4 5]]

[0 1]

0
"""

for row in X:
    print(row)

"""
출력
[0 1]
[2 3]
[4 5]
"""

X= X.flatten() # X를 1차원 배열로 변환(평탄화)
print(X)
print( X[np.array([0, 2, 4])] ) #인덱스가 0,2,4인 원소 얻기

"""
출력
[0 1 2 3 4 5]
[0 2 4]
"""


# matplotlib
print("=======matplotlib=========")

#단순한 그래프 그리기

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1간격으로 생성
y = np.sin(x)

#plt.plot(x,y)
#plt.show()

# pyplot의 기능

x = np.arange(0 , 6 ,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1, label="sine")
plt.plot(x,y2, linestyle="--", label="cos") #cos함수는 점선으로 그리기

plt.xlabel("x-axis") #x,y축 이름 설정
plt.ylabel("y-axis")

plt.title("sin & cos") #제목 설정

plt.legend() #더 알아보기
plt.show()

#이미지 표시하기
from matplotlib.image import imread


img = imread('p2.png')
plt.imshow(img)
plt.show()
