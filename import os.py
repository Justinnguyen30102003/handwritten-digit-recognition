import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from keras import models 

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Resize images to 28x28
train_images_resized = []
for img in train_images:
    resized_img = cv2.resize(img, (28, 28))
    train_images_resized.append(resized_img)
train_images_resized = np.array(train_images_resized)

test_images_resized = []
for img in test_images:
    resized_img = cv2.resize(img, (28, 28))
    test_images_resized.append(resized_img)
test_images_resized = np.array(test_images_resized)

# Flatten the images
train_images_flattened = train_images.reshape((train_images.shape[0], -1))
test_images_flattened = test_images.reshape((test_images.shape[0], -1))

# Normalize the images to the range [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Tạo ma trận trọng số và bias ngẫu nhiên
weight_matrix = np.random.rand(784, 32)
bias_matrix =np.random.rand(1, 32)

# Tính y
y = np.dot(train_images_flattened, weight_matrix) + bias_matrix
print (f'y shape: {y.shape}')

y_test = np.dot(test_images_flattened, weight_matrix) + bias_matrix
print(f'y_test shape: {y_test.shape}')

# Áp dụng hàm Leaky ReLU cho đầu ra dự đoán
alpha = 0.01
y = np.where(y > 0, y, y * alpha)

# Tạo ma trận trọng số và bias ngẫu nhiên cho lớp thứ hai
weight1_matrix = np.random.rand(32, 10)
bias1_matrix = np.random.rand(1, 10)

# Tính y1 cho tập train qua lớp thứ hai
y1 = np.dot(y, weight1_matrix) + bias1_matrix
print(f'y1 shape: {y1.shape}')

# Tính y1 cho tập test qua lớp thứ hai
y1_test = np.dot(y_test, weight1_matrix) + bias1_matrix
print(f'y1_test shape: {y1_test.shape}')

# Hàm sigmoid
def sigmoid(y1):
    return 1 / (1 + np.exp(-y1))
print(sigmoid(y1))
print(y1.shape)

def sigmoid(y1_test):
    return 1 / (1 + np.exp(-y1_test))
print(sigmoid(y1_test))
print(y1_test.shape)

# Đạo hàm hàm sigmoid
def sigmoid_derivative(y1):
    return y1*(1-y1)
print(sigmoid(y1))
print(y1.shape)
    
def sigmoid_derivative(y1_test):
    return y1_test*(1-y1_test)
print(sigmoid(y1_test))
print(y1_test.shape)

# Lớp neural network
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        
        # Hệ số learning rate
        self.alpha = alpha 
        
        #Tham số W,b
        self.W = []
        self.b = []

        # Khởi tạo các tham số ở mỗi layer
        for i in range(0, len(layers)-1):
                w_ = np.random.randn(layers[i], layers[i+1])
                b_ = np.zeros((layers[i+1], 1))
                self.W.append(w_/layers[i])
                self.b.append(b_)

    # Tóm tắt mô hình neural network
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
    
    # Train mô hình với dữ liệu
    def fit_partial(self, x, y):
        A = [x]

        # Qúa trình feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        
        # Qúa trình backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []

        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
    
    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss{}".format(epoch, loss))

    # Dự đoán
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X
    
    # Tính loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        #return np.sum((y_predict-y)**2)/2
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))




    







    





