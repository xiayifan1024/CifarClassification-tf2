import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model

class ResNetBlock(Model):
    def __init__(self, filter, strides=1, residual_path=False):
        super(ResNetBlock,self).__init__()
#        self.filters = filter
#        self.strides = strides
        self.res_path = residual_path

        self.conv_1 = Conv2D(filter, (3, 3), strides=strides, padding='same', use_bias=False) # 当卷积层后跟BN层时最好设为False
        self.bn_1 = BatchNormalization()
        self.act_1 = Activation('relu')

        self.conv_2 = Conv2D(filter, (3, 3), strides=1, padding='same', use_bias=False)
        self.bn_2 = BatchNormalization()

        if residual_path: # 对于残差模型F(x)=H(x)+Wx, 判断Hx与x是否维度一致, 如果不一致利用1*1卷积W(x),来使大小一致 
            self.down_conv_1 = Conv2D(filter, (1,1), strides=strides, padding='same', use_bias=False)
            self.down_bn_1 = BatchNormalization()
        
        self.act_2 = Activation('relu')

    def call(self,inputs):
        residual = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act_1(x)
        
        x = self.conv_2(x)
        y = self.bn_2(x)

        if self.res_path:
            residual = self.down_conv_1(inputs)
            residual = self.down_bn_1(residual)
        
        out = self.act_2(y+residual)
        return out
    
class SimplyResNet(Model):
    def __init__(self, block_list, initial_filters=64): # block_list表示每个Block有几个ResNetBlock
        super(SimplyResNet,self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.conv_1 = Conv2D(self.out_filters, (3, 3),strides=1, padding='same', use_bias=False)
        self.bn_1 = BatchNormalization()
        self.act_1 = Activation('relu')

        self.blocks = tf.keras.models.Sequential()       
        for block_id in range(len(block_list)):
            for Res_id in range(block_list[block_id]):
                if block_id !=0 and Res_id == 0: # 除了第一个block,其他block的第一个ResNetBlock的Hx与x是不一致的(模型)
                    block = ResNetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResNetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2


        self.pool = GlobalAveragePooling2D()
        
        self.fc = Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
    
    def call(self, inputs): 
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.blocks(x)
        x = self.pool(x)
        y = self.fc(x)
        return y

