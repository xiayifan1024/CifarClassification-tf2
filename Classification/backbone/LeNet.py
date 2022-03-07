import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_1 = Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv_2 = Conv2D(filters=16, kernel_size=(5, 5),
                         activation='sigmoid')
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.pool_2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y