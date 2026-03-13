# WING Survival Program

## TASK 1 : CNN 구현
### Datasets
- Source : Kaggle의 mnist-in-csv 사용 (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- 28x28 흑백 이미지 총 1만장  
- 0: 980 / 1: 1135 / 2: 1032 / 3: 1010 / 4: 982 / 5: 892 / 6: 958 / 7: 1028 / 8: 974 / 9: 1009  

### Structure
- LeNet-5 구조 모방  

- Convolution (5, 5, 6)  
- AvgPooling (2, 2)  
- Convolution (5, 5, 16)  
- AvgPooling (2, 2)  
- fc (120)  
- fc (84)  
- fc (10)  
