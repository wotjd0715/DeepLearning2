## 신경망 학습에 쓰이는 기술들
이번에는 신경망 학습의 개념에 대해 배워보자   
가중치 매개변수의 최적값을 탐색하는 최적화 방법   
가중치 매개변수 초기값   
하이퍼 파라미터 설정방법 등..   

### 매개변수 갱신
신경망 학습의 목표는 loss를 최소화 하는 매개변수(가중치,편향)을 찾는것이며 이는 곳 
매개변수값을 `최적화` 하는 것이다.   
이러한 최적화를 위해 5장에서 우리가 사용한 방벅이 `확률적 경사하강법`(SGD)이다.   
SGD를 통해 매개변수의 기울기를 통해 기울어진 방향으로 최적화 하였다.   
하지만 이는 단점이 존재한다.

- SGD의 단점   
평면의 모양에 따라서 에선 기울기가 애매해진다 -> 학습이 애매해진다

### 모멘텀
W <- W + av -lr*dL/dW  기존의 가중치 학습에서 av라는 모멘텀을 추가했음
이로써 기울기가 0인곳에서 약간의 속도를 얻어 효율적으로 backpropagation을 통해 학습할수 있다.

### AdaGrad
학습을 진행할떄 learning_rate를 정하는건 아주 중요하다.   
learning_rate가 너무 작으면 학습시간이 길어지고 너무 크면 발산하게 되어 학습이 제대로
이루어 지지 않는다.

이를 위해 나온 방법이 학습률을 서서히 감소시키는 `학습률감소`방법이 있다.
여기서 AdaGrad는 각 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행한다.
h <- h + dL/dW * dL/dW
W <- W - lr*(1/√H)*dL/dW
여기사 1/√H 를 통해 학습률을 조정할수 있다.

하지만 H는 과거의 기울기를 제곱하여 계속 더해가므로 학습이 오래 지속되면 기울기가
매우 작아져 어느순간 h가 더이상 갱신되지 않는다.

### Adam
Adam은 모멘텀과 AdaGrad의 개념을 합친 개념이다.

## 가중치의 초기값
오버피팅을 억제해 범용성능을 높이는 기술인 `가중치 감소`에 대해 알아보자    
가중치 감소란 쉽게 가중치 매개변수 값이 작아지도록 학습하는 방법이다.

가중치 값이 작아지도록할려면 초기값을 작은값에서 시작하는게 유리하므로
이제까지는 `0.01*random수`로 초기값을 설정하였다.

그렇다면 만약 초기값을 모두 0으로 설정하거나 같은 값으로 하게되면 어떨까?   
답은 학습이 제대로 이루어 지지 않는다. 그 이유는 학습의 원리가 backpropagation이기
때문이다. 만약 모든 가중치의 값이 같다면 그 미분값 역시 모두 동일하기에 모든 가중치들이
똑같이 학습된다. 따라서 이렇게 될경우 가중치를 1000개를 만들어도 1개만 있는것과 같은 상황이 된다.

## 배치 정규화
가중치의 초기값을 랜덤으로 적절히 분배하면 각 층의 활성화 값이 고르게 분포 한다.   
그렇다면 강제로 고르게 분포하도록 하면 어떨까?   
그 방벙이 `배치 정규화`이다.   
`배치정규화`를 이용하면 학습시간이 짧아지며 초기값에 크게 의존하지 않는다. 또한 오버피팅까지 억제하여 준다.

## 오버피팅
오버피팅(overfitting)이란 학습할때 훈련데이터에 너무 완벽히 학습되어 그 외 데이터는 받아들이지
못하는 상태를 말한다.   
주로 
1. 매개변수가 많고 표현력이 높은 모델(층이 깊고) 
2. 훈력데이터가 적을떄   

자주 오버피팅이 발생한다.

오버피팅을 방지하는 방법으로는 `가중치 감소` 가 있다.   
이는 말 그대로 학습하는 과정에서 큰 가중치 값에 대해서는 큰 페넬티를 부과해 오버피팅을 억제하는 방법이다.
원래 오버 피팅은 가중치 매개변수의 값이커서 발생하는 경우가 많기 떄문이다. 

두번째 방법은 `드롭아웃`이다.   
이전의 신경망은 각 노드들끼리 모두 가중치로 연결되있지만 드롭아웃을 사용하면  이 연결을
무작위 적으로 삭제한다.

## 적절한 하이퍼파라미터 값 찾기
하이퍼 파라미터란 사람이 직접 설정해주는 각층의 뉴런수, 배치크기 , 매개변수 갱신시 learning_rate, 가중치 감소 등을 말한다.
이 값들을 효율적으로 설정하는 방법을 알아보자

1.검증데이터
하이퍼 파라미터의 값을 검증할때는 시험 데이터를 사용해선 안된다. 시험데이터를 사용할경우 시험데이터에 오버피팅되므로 
다른 시험데이터가 왔을때는 적절하지 못할수 있다.   
따라서 하이퍼 파라미터를 조정할 때는 `검증데이터`라고 하는 하이퍼 파라미터 검증용 데이터를 사용한다.    
이제 우리가 데이터를 받아오고 나눌때는 3종류로 나누어 사용한다.
+ 훈련데이터: 매개변수 학습
+ 검증데이터: 하이퍼 파라미터 학습
+ 시험데이터: 신경망의 범용 성능 평가

2.하이퍼파라미터 최적화
하이퍼파라미터를 최적화 할때 핵심은 '최적값'이 존재하는 범위를 줄여나가는 거이다.
우선 대략적인 범뮈를 설정하고 그뒤 조금씩 걸러내 간다.


