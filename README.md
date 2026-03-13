# WING Survival Program

## TASK 1 : CNN 구현
### Datasets
Source : Kaggle의 mnist-in-csv 사용 (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
28x28 흑백 이미지 총 1만장  
0: 980 / 1: 1135 / 2: 1032 / 3: 1010 / 4: 982 / 5: 892 / 6: 958 / 7: 1028 / 8: 974 / 9: 1009  

#

### Structure
LeNet-5 구조 모방  

Convolution (5x5, 6 = 필터 개수) -> ReLU  
AvgPooling (2x2)  
Convolution (5x5, 16) -> ReLU  
AvgPooling (2x2)  
fc (120) -> ReLU  
fc (84) -> ReLU  
fc (10) -> ReLU  
Softmax  

#

이미지 shape 변화  

(1, 28, 28)  
(6, 24, 24)  
(6, 12, 12)  
(16, 8, 8) -> flatten  
(256)  
(120)  
(84)  
(10)  

#

weights 44,190  
bias 236  
total 44,426 paramerters  

#

### Forward propagation

$$y_{f,i,j} = \sum_{c=1}^{C} \sum_{u=1}^{k} \sum_{v=1}^{k}
w_{f,c,u,v} \cdot x_{c,i+u-1,j+v-1} + b_f$$

Convolution Layer 수식  
$$f$$ : 아웃풋 채널  
$$i, j$$ : 픽셀 $$y$$의 위치  
$$C$$ : 인풋 채널  
$$k$$ : 커널 사이즈(5)  
$$w_{f,c,u,v}$$ : $$f$$번째 아웃풋 채널, $$c$$번째 인풋 채널, $$[u, v]$$칸의 가중치  
$$x$$ : 인풋 행렬  
$$b_f$$ : $$f$$번째 채널의 편향  

#

$$\mathrm{ReLU}(x) = \max(0, x)$$

ReLU 활성화 함수

#

$$y_{i,j} =
\frac{1}{k^2}
\sum_{u=1}^{k}\sum_{v=1}^{k}
x_{k i + u, k j + v}$$

Avgerage Pooling Layer 수식
$$k$$ : 커널 사이즈(2)

#

$$z = Wx + b$$

Fully Connected Layer 수식
$$W$$ : 가중치
$$b$$ : 편향

#

$$p_i =
\frac{e^{z_i}}
{\sum_{j=1}^{C} e^{z_j}}$$

Softmax 수식

#

### Train (backpropagation)
손실은 Cross Entropy Loss 사용  

**Cross Entropy Loss**\
$$L = - \log(p_y)$$

$$p_y$$는 정답 클래스 y에 대한 확률
