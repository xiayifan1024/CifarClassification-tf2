import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
# LRN目前主流算法不使用,改为BN
class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv_1 = Conv2D(filters = 96, kernel_size = (3, 3))
        self.bn_1 = BatchNormalization()
        self.act_1 = Activation('relu')
        # 先进行BN再激活:https://www.zhihu.com/question/318354788
        self.pool_1 = MaxPool2D(pool_size = (3, 3), strides = 2)

        self.conv_2 = Conv2D(filters = 256, kernel_size = (3, 3))
        self.bn_2 = BatchNormalization()
        self.act_2 = Activation('relu')
        self.pool_2 = MaxPool2D(pool_size = (3, 3), strides = 2)

        self.conv_3 = Conv2D(filters = 384, kernel_size = (3, 3), activation = 'relu', padding = 'same' )
        
        self.conv_4 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same' )
        self.pool_3 = MaxPool2D(pool_size = (3,3), strides = 2)

        self.flatten = Flatten()
        self.fc_1 = Dense(2048,activation='relu')
        self.drop_1 = Dropout(0.5)
        self.fc_2 = Dense(2048,activation='relu')
        self.drop_2 = Dropout(0.5)
        self.fc_3 = Dense(10,activation='softmax')

    def call(self, inputs):
        x = self.conv_1(inputs) 
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        
        x = self.conv_4(x)
        x = self.pool_3(x)

        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        y = self.fc_3(x)
        return y