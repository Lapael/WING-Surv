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
Backpropagation에 대해서는 기존에 수학적으로 이해하고 있지 않았기 때문에,  
좀 더 엄밀하게 서술하도록 하겠다.  

손실은 Cross Entropy Loss 사용  

$$L = - y_i\log(p_y)$$

Cross Entropy Loss 수식  
$$p_y$$ : 클래스 y에 대한 예측  
$$y_i$$ : y가 정답 클래스면 1, 아니면 0  

-> 정답 클래스에 대한 값만 역전파에 사용됨.  

#

역전파 = Loss 값을 기반으로 gradient를 계산하는 것.  
이렇게 계산한 gradient를 기반으로 optimizer을 이용해 파라미터를 업데이트해서 학습  

#

#### Prob -> Softmax 역전파

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial z_j}$$

$$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$$

$$\frac{\partial p_i}{\partial z_j} = \frac{(e^{z_i})'\sum e^{z_j} - e^{z_i} (\sum e^{z_j})'}{(\sum e^{z_j})^2}$$

$$z_j$$에 대해 미분하는 과정이기에 $$z_i (i \neq j)$$는 다른 변수이기에 미분할 시 0이 된다.  
이를 기호로 표현한 게 크로네커 델타(Kronecker delta, $$\delta_{ij}$$)라고 한다.  

$$
\delta_{ij} =
\begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
$$

따라서 앞서 몫 미분으로 표현한 식을 계산하기 위해 분자, 분모를 각각 미분하면,

$$\frac{\partial}{\partial z_j}e^{z_i} = \delta_{ij} e^{z_i}$$

$$\frac{\partial \sum e^{z_j}}{\partial z_j} = e^{z_j}$$

이에 따라

$$\frac{\partial p_i}{\partial z_j} = \frac{(\delta_{ij} e^{z_i})\sum e^{z_j} - e^{z_i}e^{z_j}}{(\sum e^{z_j})^2}$$

$$\frac{\delta_{ij}e^{z_i}}{\sum e^{z_j}} = p_i delta_{ij}$$

$$\frac{e^{z_i}e^{z_j}}{(\sum e^{z_j})^2} = p_i p_j$$

$$\frac{\partial p_i}{\partial z_j} = p_i \delta_{ij} - p_i p_j$$

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial z_j} = \sum_i (-\frac{y_i}{p_i})(p_i \delta_{ij} - p_i p_j)$$ = -y_j + \sum_i y_i p_j = p_j - y_j$$

$$\frac{\partial L}{\partial z_j} = p_j - y_j = p_j - 1$$

즉, prob 값에서 softmax 이전(fc10 이후)으로 역전파하면 정답 클래스의 prob값에서 1을 빼면 된다.  
이 부분이 utils.py의 class Softmax 하위의 backward 함수에 구현되어있다.  


#### Softmax -> fc 역전파

fc layer의 수식 :

$$z = Wx + b$$

여기서 업데이트해야 하는 값은 $$W$$와 $$b$$,  
각 변수의 shape :  
$$x$$ : (D)  
$$W$$ : (C, D)  
$$b$$ : (C)  
$$z$$ : (C)  

D = 입력 feature 수, C = 출력 feature 수

$$z_i = \sum_{j=1}^D W_{ij}x_j + b_i$$

가 된다. 또한 마지막에 얻어야 하는 값들은  

$$\frac{\partial L}{\partial W}$$

$$\frac{\partial L}{\partial b}$$

$$\frac{\partial L}{\partial x}$$

이고  

$$\frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial W_{ij}} = \frac{\partial L}{\partial W_{ij}}$$

$$\frac{\partial z_i}{\partial W_{ij}} = x_j$$

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} x_j$$

$$\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial z}) x^T$$

이 된다. (shape도 (C, D)로 동일해짐)  

$$z = Wx + b$$

$$\frac{\partial z}{\partial b} = 1$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$$

이 된다. (shape도 (C))

$$z_i = \sum_{j=1}^D W_{ij}x_j + b_i$$

$$\frac{\partial z_i}{\partial x_j} = W_{ij}$$

$$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} W_{ij}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} W^T$$

이 된다. (shape는 (D))

결국 얻어야 했던 것들을 모두 계산했다.

$$\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial z}) x^T$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} W^T$$

여기서 $$\frac{\partial L}{\partial x}$$는 이전 레이어로 전달되고  
$$\frac{\partial L}{\partial W}$$와 $$\frac{\partial L}{\partial b}$$는 optimizer가 파라미터를 업데이트 할 때 쓰인다.

### fc -> AvgPool 역전파

AvgPooling Layer은 단순 다운샘플링 레이어이기 때문에  
gradient($$\frac{\partial L}{\partial y}$$)를 동등하게 분배하여 역전파한다.

