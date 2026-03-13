import csv
import cv2
import numpy as np
import os

r = csv.reader(open('mnist_test.csv', 'r'))

num = [0,0,0,0,0,0,0,0,0,0]

for i in range(10):
    if not os.path.isdir(f'{i}'):
        os.mkdir(f'{i}')

for x in r:
    idx = int(x[0])
    num[idx] += 1
    img = np.reshape(np.delete(np.array(x, dtype=np.uint8),0),(28, 28))
    print(idx)
    cv2.imwrite(f'{idx}\\{num[idx]}.jpg', img)

print(num)

# source : https://www.kaggle.com/datasets/oddrationale/mnist-in-csv