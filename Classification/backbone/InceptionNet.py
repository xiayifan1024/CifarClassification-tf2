import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
# https://jaketae.github.io/study/pytorch-inception/
# 理解1*1卷积的作用
# 卷积模块
class ConvBlock(Model):
    def __init__(self, channel, kenelsize=3, strides=1, padding='same'):
        super(ConvBlock,self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(channel,kernel_size=kenelsize, strides=strides,padding=padding,use_bias=False),
            BatchNormalization(),
            Activation('relu')
        ])
    
    def call(self,inputs):
        y = self.model(inputs)
        return y
'''
# 简化Inception模块
class NaiveInceptionBlock(Model):
    def __init__(self,channel,strides=1):
        super(NaiveInceptionBlock,self).__init__()
        self.channel = channel
        self.strides = strides
        self.conv_1 = ConvBlock(channel, kenelsize=1, strides=1)
        self.conv_3 = ConvBlock(channel,kenelsize=3,strides=1)
        self.conv_5 = ConvBlock(channel,kenelsize=5,strides=1)
        self.pool = MaxPool2D(3,strides=1,padding='same')
    
    def call(self,inputs):
        x1 = self.conv_1(inputs)
        x2 = self.conv_3(inputs)
        x3 = self.conv_5(inputs)
        x4 = self.pool(inputs)

        y = tf.concat([x1,x2,x3,x4],axis=3)
        return y
'''
# InceptionV1模块
class InceptionBlock(Model):
    def __init__(self,channel,strides=1):
        super(InceptionBlock,self).__init__()
#        self.channel = channel
#        self.strides = strides
        self.conv_1 = ConvBlock(channel, kenelsize=1, strides=strides)
        self.conv_3 = ConvBlock(channel,kenelsize=3,strides=1)
        self.conv_5 = ConvBlock(channel,kenelsize=5,strides=1)
        self.pool = MaxPool2D(3,strides=1,padding='same')
    
    def call(self,inputs):   
        x1 = self.conv_1(inputs)        
        x2 = self.conv_3(x1) 
        x3 = self.conv_5(x1)
        x4_ = self.pool(inputs)
        x4 = self.conv_1(x4_)
        y = tf.concat([x1,x2,x3,x4],axis=3) # 延深度方向堆叠,与ResNet数值相加区别
        return y
# 简易InceptionNet
class SimplyInceptionNet(Model):
    def __init__(self, num_block, num_classes, init_ch=16, **kwargs):
        super(SimplyInceptionNet, self).__init__(**kwargs) # https://www.jianshu.com/p/0ed914608a2c **kwargs的作用
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_block
        self.conv_1 = ConvBlock(init_ch) 
        self.blocks = tf.keras.models.Sequential()
        for block_id in range (num_block):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels,strides=2)
                else:
                    block = InceptionBlock(self.out_channels,strides=1)
                self.blocks.add(block)
            
        self.out_channels *= 2
        self.pool = GlobalAveragePooling2D()
        self.fc =  Dense(num_classes,activation='softmax')

    def call(self,inputs):
        x = self.conv_1(inputs)
        x = self.blocks(x)
        x = self.pool(x)
        y = self.fc(x)
            
        return y