import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pandas as pd
from keras import models 

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('D:/handwritten.keras')
# model = tf.keras.models.load_model('D:/handwritten.keras')
# model = tf.keras.layers.TFSMLayer(handwritten.model, call_endpoint = 'serving_default')


image_number = 1
print(os.path.isfile(f"D:/Praktikum/Handwritten Digit Recognition/digits/digit{image_number}.png"))
while os.path.isfile(f"D:/Praktikum/Handwritten Digit Recognition/digits/digit{image_number}.png"):
    img = cv2.imread(f"D:/Praktikum/Handwritten Digit Recognition/digits/digit{image_number}.png")
    img = np.invert(np.array([img]))
    img.resize((1, 28, 28))
    flattened_img = img.reshape(784, 1)
    transposed_flattened = flattened_img.T
    weight_matrix = np.random.rand(784, 32)
    bias_matrix =np.random.rand(1, 32)
    y = np.dot(transposed_flattened, weight_matrix) + bias_matrix
    print (y.shape)


    # prediction = model.predict(img)
    alpha = 0.01
    y = np.where(y > 0, y, y * alpha)
    
    weight1_matrix = np.random.rand(32, 10)
    bias1_matrix = np.random.rand(1, 10)
    y1 = np.dot(y, weight1_matrix) + bias1_matrix
    print(y1.shape)

   # Hàm sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    print(sigmoid(y1))
    print(f"This digit is probably a {np.argmax(y1)}")

    # Đạo hàm hàm sigmoid
    def sigmoid_derivative(x):
        return x*(1-x)
    print(sigmoid(y1))
    print(f"This digit is probably a {np.argmax(y1)}")

    def number_to_one_hot(number, num_classes):
        one_hot = np.zeros(num_classes)  # Initialize all elements to 0
        one_hot[number] = 1  # Set the element corresponding to the number to 1
        return one_hot
    one_hot_encoded = number_to_one_hot(number = 1, num_classes = 10)
    print(one_hot_encoded)
    
    def mean_squared_error(y_true, y_pred):
        squared_error = np.square(y_true - y_pred)
        mse = np.mean(squared_error)
        return mse
    y_true = one_hot_encoded
    y_pred = y1
    loss = mean_squared_error(y_true, y_pred)

    # Lớp neural network
    class NeuralNetwork: 
        def __init__(self, layers, alpha=0.1):
            self.layers = layers
                # He so learning rate
            self.alpha = alpha 
                # Tham số W, b
            self.W = []
            self.b = []  



    # Khởi tạo các tham số ở mỗi layer


    # Tóm tắt mô hình neural network
    def __repr__(self):
        return "Neural network [{}]" .format("-".join(str(l) for l in self.layers))

    # Train mô hình với dữ liệu
    def fit_partial(self, x, y):
        A = [x]

        # quá trình feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)

        # quá trình backpropagation
        y = y.reshape(-1,1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers)-1)):
            dW_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dW_)
            db.append(db_)
            dA.append(dA_)

        # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]
        

                # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
    def fit(self, X, y, epochs=10, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))

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
    print("Mean Squared Error:", loss)

    img = np.array(img)
    plt.imshow(img.T)
    plt.show()
    plt.close()

    image_number +=1






