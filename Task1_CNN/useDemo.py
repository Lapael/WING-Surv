from utils import *
import numpy as np
import cv2

AvgPool0 = AvgPool10x10()
conv1 = ConvNxN(5, 6,  1)
conv2 = ConvNxN(5, 16, 6)
AvgPool1 = AvgPool2x2()
AvgPool2 = AvgPool2x2()
fc120 = fc(120, 256)
fc84 = fc(84, 120)
fc10 = fc(10, 84, False)
sm = Softmax()

params = np.load('model_params_2.npz')
conv1.filter_matrix = params['conv1_filter']
conv1.bias = params['conv1_bias']
conv2.filter_matrix = params['conv2_filter']
conv2.bias = params['conv2_bias']
fc120.weights = params['fc120_weights']
fc120.biases = params['fc120_biases']
fc84.weights = params['fc84_weights']
fc84.biases = params['fc84_biases']
fc10.weights = params['fc10_weights']
fc10.biases = params['fc10_biases']

oldx = oldy = -1
oldx_r = oldy_r = -1

def use(img):
    img = AvgPool0.forward(img)
    x = conv1.forward(img / 255)
    x = AvgPool1.forward(x)
    x = conv2.forward(x)
    x = AvgPool2.forward(x)
    x = fc120.forward(x)
    x = fc84.forward(x)
    x = fc10.forward(x)
    x = sm.forward(x)
    x *= 100
    x = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
    print('\n\n\n\n\n\n')
    for label, prob in x[0:5]:
        print(f'{label}: {round(prob, 2)}%')

def on_mouse(event, x, y, flags, param):
    global oldx, oldy, oldx_r, oldy_r

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        pass # forward

    elif event == cv2.EVENT_RBUTTONDOWN:
        oldx_r, oldy_r = x, y

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        cv2.rectangle(img, (0, 0), (279, 279), (0, 0, 0), -1)
        cv2.imshow('Demo', img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 15, cv2.LINE_AA)
            cv2.imshow('Demo', img)
            oldx, oldy = x, y
            img_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = img_np[np.newaxis, :, :]
            use(img_np)


        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.line(img, (oldx_r, oldy_r), (x, y), (0, 0, 0), 20, cv2.LINE_AA)
            cv2.imshow('Demo', img)
            oldx_r, oldy_r = x, y
            img_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = img_np[np.newaxis, :, :]
            use(img_np)

img = np.zeros((280, 280, 3), dtype=np.uint8)

cv2.namedWindow('Demo')

cv2.setMouseCallback('Demo', on_mouse, img)

cv2.imshow('Demo', img)
cv2.waitKey()

cv2.destroyAllWindows()