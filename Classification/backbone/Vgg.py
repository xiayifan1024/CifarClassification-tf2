import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
# 这里的vgg使用了BatchNormalization
# https://blog.csdn.net/lyl771857509/article/details/84175874
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        # block_1
        # 后记:可以通过循环和将conv,bn,act三层封装成类对代码进行简化
        # conv_1
        self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.bn_1 = BatchNormalization()  # BN层1
        self.act_1 = Activation('relu')  # 激活层1
        # conv_2
        self.conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.bn_2 = BatchNormalization()  # BN层1
        self.act_2 = Activation('relu')  # 激活层1
        # Maxpool_1
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d1 = Dropout(0.2)  
        # https://zhuanlan.zhihu.com/p/61725100 说明了Dropout层与BN层不需要并用
        # block_2
        # conv_3
        self.conv_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn_3 = BatchNormalization()  # BN层1
        self.act_3 = Activation('relu')  # 激活层1
        # conv_4
        self.conv_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn_4 = BatchNormalization()  # BN层1
        self.act_4 = Activation('relu')  # 激活层1
        # Maxpool_2
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d2 = Dropout(0.2)  # dropout层
 
        # block_3
        # conv_5
        self.conv_5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_5 = BatchNormalization()  # BN层1
        self.act_5 = Activation('relu')  # 激活层1
        # conv_6
        self.conv_6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_6 = BatchNormalization()  # BN层1
        self.act_6 = Activation('relu')  # 激活层1
        # conv_7
        self.conv_7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_7 = BatchNormalization()
        self.act_7 = Activation('relu')
        # Maxpool_3
        self.pool_3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d3 = Dropout(0.2)
 
        # block_4
        # conv_8
        self.conv_8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_8 = BatchNormalization()  # BN层1
        self.act_8 = Activation('relu')  # 激活层1
        # conv_9
        self.conv_9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_9 = BatchNormalization()  # BN层1
        self.act_9 = Activation('relu')  # 激活层1
        # conv_10
        self.conv_10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_10 = BatchNormalization()
        self.act_10 = Activation('relu')
        # Maxpool_4
        self.pool_4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d4 = Dropout(0.2)
 
        # block_5
        # conv_11
        self.conv_11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_11 = BatchNormalization()  # BN层1
        self.act_11 = Activation('relu')  # 激活层1
        # conv_12
        self.conv_12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_12 = BatchNormalization()  # BN层1
        self.act_12 = Activation('relu')  # 激活层1
        # conv_13
        self.conv_13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_13 = BatchNormalization()
        self.act_13 = Activation('relu')
        # Maxpool_5
        self.pool_5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        #self.d5 = Dropout(0.2)

        # FullyConnect
        self.flatten = Flatten()
        self.fc_1 = Dense(512, activation='relu')
        self.d_1 = Dropout(0.2)
        self.fc_2 = Dense(512, activation='relu')
        self.d_2 = Dropout(0.2)
        self.fc_3 = Dense(10, activation='softmax')
 
    def call(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        
        x = self.pool_1(x)
        # x = self.d1(x)
 
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)
        
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act_4(x)
        x = self.pool_2(x)
        # x = self.d2(x)
 
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act_5(x)
        
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.act_6(x)
        
        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.act_7(x)
        
        x = self.pool_3(x)
        # x = self.d3(x)
 
        x = self.conv_8(x)
        x = self.bn_8(x)
        x = self.act_8(x)
        
        x = self.conv_9(x)
        x = self.bn_9(x)
        x = self.act_9(x)
        x = self.conv_10(x)
        x = self.bn_10(x)
        x = self.act_10(x)
        x = self.pool_4(x)
        # x = self.d4(x)
 
        x = self.conv_11(x)
        x = self.bn_11(x)
        x = self.act_11(x)
        
        x = self.conv_12(x)
        x = self.bn_12(x)
        x = self.act_12(x)
        
        x = self.conv_13(x)
        x = self.bn_13(x)
        x = self.act_13(x)
        x = self.pool_5(x)
        # x = self.d5(x)
 
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.d_1(x)
        x = self.fc_2(x)
        x = self.d_2(x)
        y = self.fc_3(x)
        return y

class VGG19(Model):
    def __init__(self):
        super(VGG19, self).__init__()
        # block_1
        # conv_1
        self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.bn_1 = BatchNormalization()  # BN层1
        self.act_1 = Activation('relu')  # 激活层1
        # conv_2
        self.conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.bn_2 = BatchNormalization()  # BN层1
        self.act_2 = Activation('relu')  # 激活层1
        # Maxpool_1
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d1 = Dropout(0.2)  
        # https://zhuanlan.zhihu.com/p/61725100Dropout层与BN层不需要并用
         
        # block_2
        # conv_3
        self.conv_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn_3 = BatchNormalization()  # BN层1
        self.act_3 = Activation('relu')  # 激活层1
        # conv_4
        self.conv_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn_4 = BatchNormalization()  # BN层1
        self.act_4 = Activation('relu')  # 激活层1
        # Maxpool_2
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d2 = Dropout(0.2)  # dropout层
 
        # block_3
        # conv_5
        self.conv_5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_5 = BatchNormalization()  # BN层1
        self.act_5 = Activation('relu')  # 激活层1
        # conv_6
        self.conv_6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_6 = BatchNormalization()  # BN层1
        self.act_6 = Activation('relu')  # 激活层1
        # conv_7
        self.conv_7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_7 = BatchNormalization()
        self.act_7 = Activation('relu')
        # conv_8
        self.conv_8 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_8 = BatchNormalization()  # BN层1
        self.act_8 = Activation('relu')  # 激活层1

        # Maxpool_3
        self.pool_3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d3 = Dropout(0.2)
 
        # block_4
        # conv_9
        self.conv_9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_9 = BatchNormalization()  # BN层1
        self.act_9 = Activation('relu')  # 激活层1
        # conv_10
        self.conv_10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_10 = BatchNormalization()
        self.act_10 = Activation('relu')
        # conv_11
        self.conv_11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_11 = BatchNormalization()  # BN层1
        self.act_11 = Activation('relu')  # 激活层1
        # conv_12
        self.conv_12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_12 = BatchNormalization()  # BN层1
        self.act_12 = Activation('relu')  # 激活层1
        # Maxpool_4
        self.pool_4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d4 = Dropout(0.2)
 
        # block_5
        # conv_13
        self.conv_13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_13 = BatchNormalization()
        self.act_13 = Activation('relu')
        # conv_14
        self.conv_14 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_14 = BatchNormalization()
        self.act_14 = Activation('relu')
        # conv_15
        self.conv_15 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_15 = BatchNormalization()
        self.act_15 = Activation('relu')
        # conv_16
        self.conv_16 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn_16 = BatchNormalization()
        self.act_16 = Activation('relu')
        # Maxpool_5
        self.pool_5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        #self.d5 = Dropout(0.2)

        # FullyConnect
        self.flatten = Flatten()
        # FullyConnectLayer_1
        self.fc_1 = Dense(512, activation='relu')
        self.d_1 = Dropout(0.2)
        # FullyConnectLayer_2
        self.fc_2 = Dense(512, activation='relu')
        self.d_2 = Dropout(0.2)
        # FullyConnectLayer_3
        self.fc_3 = Dense(10, activation='softmax')
 
    def call(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        
        x = self.pool_1(x)
        # x = self.d1(x)
 
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)
        
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act_4(x)
        
        x = self.pool_2(x)
        # x = self.d2(x)
 
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act_5(x)
        
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.act_6(x)
        
        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.act_7(x)

        x = self.conv_8(x)
        x = self.bn_8(x)
        x = self.act_8(x)
        
        x = self.pool_3(x)
        # x = self.d3(x)
 
        x = self.conv_9(x)
        x = self.bn_9(x)
        x = self.act_9(x)

        x = self.conv_10(x)
        x = self.bn_10(x)
        x = self.act_10(x)
        
        x = self.conv_11(x)
        x = self.bn_11(x)
        x = self.act_11(x)

        x = self.conv_12(x)
        x = self.bn_12(x)
        x = self.act_12(x)

        x = self.pool_4(x)
        # x = self.d4(x)
        #         
        x = self.conv_13(x)
        x = self.bn_13(x)
        x = self.act_13(x)
        
        x = self.conv_14(x)
        x = self.bn_14(x)
        x = self.act_14(x)
        
        x = self.conv_15(x)
        x = self.bn_15(x)
        x = self.act_15(x)

        x = self.conv_16(x)
        x = self.bn_16(x)
        x = self.act_16(x) 

        x = self.pool_5(x)
        # x = self.d5(x)
 
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.d_1(x)

        x = self.fc_2(x)
        x = self.d_2(x)
        
        y = self.fc_3(x)
        return y
