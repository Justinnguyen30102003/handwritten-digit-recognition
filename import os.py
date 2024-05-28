import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from sklearn.metrics import accuracy_score


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

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

y_test = np.dot(test_images_flattened, weight_matrix) + bias_matrix

# Áp dụng hàm Leaky ReLU cho đầu ra dự đoán
alpha = 0.01
y = np.where(y > 0, y, y * alpha)
y_test = np.where(y_test > 0, y_test, y_test * alpha)

# Tạo ma trận trọng số và bias ngẫu nhiên cho lớp thứ hai
weight1_matrix = np.random.rand(32, 10)
bias1_matrix = np.random.rand(1, 10)

# Tính y1 cho tập train qua lớp thứ hai
y = np.dot(y, weight1_matrix) + bias1_matrix
print(f'y shape: {y.shape}')

# Tính y1 cho tập test qua lớp thứ hai
y_test = np.dot(y_test, weight1_matrix) + bias1_matrix
print(f'y_test shape: {y_test.shape}')

# Hàm sigmoid
def sigmoid(y):
    return 1 / (1 + np.exp(-y))
print(sigmoid(y))
print(y.shape)

def sigmoid(y_test):
    return 1 / (1 + np.exp(-y_test))
print(sigmoid(y_test))
print(y_test.shape)

train_predictions = sigmoid(y)
test_predictions = sigmoid(y_test)

# Đạo hàm hàm sigmoid
def sigmoid_derivative(y):
    return y*(1-y)
print(sigmoid(y))
print(y.shape)
    
def sigmoid_derivative(y_test):
    return y_test*(1-y_test)
print(sigmoid(y_test))
print(y_test.shape)

# Hàm tính độ chính xác
def calculate_accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# Tính độ chính xác
train_accuracy = calculate_accuracy(train_predictions, train_labels)
test_accuracy = calculate_accuracy(test_predictions, test_labels)

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

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
    def fit_partial(self, train_images, train_labels):
        A = [train_images]

        # Qúa trình feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        
        # Qúa trình backpropagation
        train_labels = train_labels.reshape(-1, 1)
        dA = [-(train_labels/A[-1] - (1-train_labels)/(1-A[-1]))]
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
    
    def fit(self, train_images, train_labels, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(train_images, train_labels)
            if epoch % verbose == 0:
                loss = self.calculate_loss(train_images, train_labels)
                print("Epoch {}, loss{}".format(epoch, loss))

    # Dự đoán
    def predict(self, train_images):
        for i in range(0, len(self.layers) - 1):
            train_images = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return train_images
    
    # Tính loss function
    def calculate_loss(self, train_images, train_labels):
        train_labels_predict = self.predict(train_images)
        #return np.sum((y_predict-y)**2)/2
        return -(np.sum(y*np.log(train_labels_predict) + (1-train_labels)*np.log(1-train_labels_predict)))
    
# Tạo và huấn luyện mô hình Neural Network
nn = NeuralNetwork([784, 32, 10])
nn.fit(train_images, train_labels, epochs=20, verbose=5)




    







    





