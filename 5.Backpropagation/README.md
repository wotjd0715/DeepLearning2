## 5.Backpropagation(오차역전파)
4장에서 신경망을 학습시킬때 가중치 매개변수의 기울기를 미분을 통해 구했음
하지만 미분은 계산 시간이 너무 오래걸린다.   
이번 5장에서는 기울기를 효율적으로 구하는 방법을 알아보자

### 오차역전파

오차역전파를 알기전에 순전파 부터 알자
2 x 5 x 1.1 = 11을 계산할때 우리는 왼쪽에서 오른쪽으로 게산을 진행하는데 이게 `순전파`다   
`역전파`는 당연하게도 오른쪽에서 왼쪽으로 계산을 진행하는 것을 뜻하며 이는 미분을 계산할때 중요한 역활을 한다.   

