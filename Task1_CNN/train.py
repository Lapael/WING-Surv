from utils import *

conv1 = ConvNxN(5, 6,  1)
conv2 = ConvNxN(5, 16, 6)
AvgPool1 = AvgPool2x2()
AvgPool2 = AvgPool2x2()
fc120 = fc(120, 256)
fc84 = fc(84, 120)
fc10 = fc(10, 84, False)
sm = Softmax()

'''
1   28  28
6   24  24
6   12  12
16  8   8
16  4   4
256         (Flatten)
120
84
10
'''

def forward(img, label):
    x = conv1.forward(img / 255)
    x = AvgPool1.forward(x)
    x = conv2.forward(x)
    x = AvgPool2.forward(x)
    x = fc120.forward(x)
    x = fc84.forward(x)
    x = fc10.forward(x)
    x = sm.forward(x)

    loss = -np.log(x[label])
    accuracy = 1 if np.argmax(x) == label else 0

    return loss, accuracy, x

def backward(label, lr):
    x = sm.backward(label)
    x = fc10.backward(x)
    x = fc84.backward(x)
    x = fc120.backward(x)
    x = AvgPool2.backward(x.reshape(16, 4, 4))
    x = conv2.backward(x)
    x = AvgPool1.backward(x)
    x = conv1.backward(x)

    conv1.filter_matrix -= lr * conv1.dfilter
    conv1.bias -= lr * conv1.dbias
    conv2.filter_matrix -= lr * conv2.dfilter
    conv2.bias -= lr * conv2.dbias

    fc10.weights -= lr * fc10.dweights
    fc10.biases -= lr * fc10.dbiases
    fc84.weights -= lr * fc84.dweights
    fc84.biases -= lr * fc84.dbiases
    fc120.weights -= lr * fc120.dweights
    fc120.biases -= lr * fc120.dbiases

def train(epochs, lr=0.01, save_model=False):

    loss = 0
    correct = 0

    data = np.loadtxt('datasets/mnist_test.csv', delimiter=',', dtype=int)

    labels = data[:, 0]
    images = data[:, 1:].reshape(-1, 28, 28)

    perm = np.random.permutation(len(images))
    labels = labels[perm]
    images = images[perm]

    for epoch in range(epochs):
        for attempt in range(len(images)-1000):
            l, acc, _ = forward(images[attempt], labels[attempt])
            backward(labels[attempt], lr)
            loss += l
            correct += acc

            if attempt % 100 == 99:
                print('Training')
                print(f'Epoch {epoch+1} Step {attempt+1} Past 100 Steps | Average Loss : '+'\u001b[31m'+f'{round(loss/100, 3)}'+'\u001b[0m, ' 'Accuracy : '+f'\u001b[92m{round(correct, 3)}'+'\u001b[0m'+'%')
                print('')

                loss = 0
                correct = 0

        for attempt in range(1000):
            l, acc, _ = forward(images[attempt+9000], labels[attempt+9000])

            loss += l
            correct += acc

        print('Validation')
        print(f'Epoch {epoch+1} Validation 1000 Steps | Average Loss : '+'\u001b[31m'+f'{round(loss/1000, 3)}'+'\u001b[0m, ' 'Accuracy : '+f'\u001b[92m{round(correct/10, 3)}'+'\u001b[0m'+'%')

        loss = 0
        correct = 0

        if save_model:
            np.savez(f'model_params_{epoch+1}', conv1_filter=conv1.filter_matrix, conv1_bias=conv1.bias, conv2_filter=conv2.filter_matrix, conv2_bias=conv2.bias, fc120_weights=fc120.weights, fc120_biases=fc120.biases, fc84_weights=fc84.weights, fc84_biases=fc84.biases, fc10_weights=fc10.weights, fc10_biases=fc10.biases)
            arr.append((round(loss/1000, 3),round(correct/10, 3)))
            print(f'model_params_{epoch+1}.npz is saved.')

arr = []

if __name__ == '__main__':
    train(3, lr=0.01, save_model=True)
    print(arr)