import numpy as np

class ConvNxN:
    def __init__(self, n, filter_num, input_ch=1):
        self.n = n                      # 필터 크기
        self.filter_num = filter_num    # 출력 채널
        self.input_ch = input_ch        # 입력 채널
        self.filter_matrix = np.random.randn(filter_num, input_ch, n, n) / 9
        self.bias = np.zeros(filter_num)

        self.dfilter = np.zeros_like(self.filter_matrix)
        self.dbias = np.zeros_like(self.bias)

    def ReLU(self, input_matrix):
        return np.maximum(0, input_matrix)

    def forward(self, input_img):
        if input_img.ndim == 2:
            input_img = input_img[np.newaxis, :, :]

        self.last_input = input_img.copy() # backpropagation

        input_ch, x, y = input_img.shape
        result = np.zeros((self.filter_num, x+1-self.n, y+1-self.n)) # -n + 1
        for f in range(self.filter_num):
            filt = self.filter_matrix[f]
            b = self.bias[f]
            for col in range(x+1-self.n):
                for row in range(y+1-self.n):
                    selected_area = input_img[:, col:col+self.n, row:row+self.n]
                    result[f, col, row] = np.sum(selected_area * filt) + b

        self.last_result = result
        self.last_output = self.ReLU(result) # backpropagation

        return self.last_output

    def backward(self, d_out):
        in_ch, x, y = self.last_input.shape
        out_x, out_y = d_out.shape[1], d_out.shape[2]

        self.dfilter.fill(0)
        self.dbias.fill(0)

        d_result = d_out * (self.last_result > 0)

        for f in range(self.filter_num):
            self.dbias[f] = np.sum(d_result[f])

        d_input = np.zeros_like(self.last_input)
        for f in range(self.filter_num):
            filt = self.filter_matrix[f]
            for col in range(out_x):
                for row in range(out_y):
                    patch = self.last_input[:, col:col+self.n, row:row+self.n]
                    grad_val = d_result[f, col, row]
                    self.dfilter[f] += grad_val * patch
                    d_input[:, col:col+self.n, row:row+self.n] += grad_val * filt
        return d_input


class AvgPool2x2:
    def forward(self, input_img):

        self.last_input_shape = input_img.shape # backpropagation

        input_ch, x, y = input_img.shape
        result = np.zeros((input_ch, x//2, y//2))
        for f in range(input_ch):
            for col in range(x//2):
                for row in range(y//2):
                    selected_area = input_img[f, 2*col:2*col+2, 2*row:2*row+2]
                    result[f, col, row] = np.average(selected_area)
        return result

    def backward(self, d_out):
        ch, x, y = self.last_input_shape
        d_input = np.zeros(self.last_input_shape)
        for f in range(ch):
            for col in range(d_out.shape[1]):
                for row in range(d_out.shape[2]):
                    grad = d_out[f, col, row] / 4
                    d_input[f, 2*col:2*col+2, 2*row:2*row+2] += grad
        return d_input

class AvgPool10x10:
    def forward(self, input_img):
        input_ch, x, y = input_img.shape
        result = np.zeros((input_ch, x//10, y//10))
        for f in range(input_ch):
            for col in range(x//10):
                for row in range(y//10):
                    selected_area = input_img[f, 10*col:10*col+10, 10*row:10*row+10]
                    result[f, col, row] = np.average(selected_area)
        return result

class fc:
    def __init__(self, n, input_dim, use_relu=True):
        self.n = n # 출력 벡터 차원
        self.weights = np.random.randn(n, input_dim) / n
        self.biases = np.zeros(n)
        self.use_relu = use_relu

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def ReLU(self, input_matrix):
        return np.maximum(0, input_matrix)

    def forward(self, input):
        input = input.flatten()

        self.last_input = input # backpropagation

        result = np.zeros(self.n)
        for f in range(self.n):
            result[f] = np.sum(self.weights[f] * input) + self.biases[f]

        self.last_result = result # backpropagation

        if self.use_relu:
            self.last_output = self.ReLU(result)
        else:
            self.last_output = result # backpropagation

        return self.last_output

    def backward(self, d_out):
        if self.use_relu:
            d_result = d_out * (self.last_result > 0)
        else:
            d_result = d_out.copy()

        self.dweights = d_result.reshape(-1, 1) * self.last_input.reshape(1, -1)
        self.dbiases = d_result

        d_input = np.dot(self.weights.T, d_result)
        return d_input.reshape(self.last_input.shape)


class Softmax:
    def forward(self, input):

        self.last_logits = input # backpropagation

        total = np.sum(np.exp(input))
        self.last_result = np.exp(input) / total # backpropagation
        return self.last_result

    def backward(self, label):
        dlogits = self.last_result.copy()
        dlogits[label] -= 1
        return dlogits
