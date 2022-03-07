# 参考https://www.bilibili.com/video/BV1B7411L7Qt?p=25
import os
# 出现Function call stack:predict_function 报错解决方法1
# 用如下代码不使用GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# 或用如下代码选择GPU：
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from PIL import Image
import numpy as np
import tensorflow as tf
# 出现Function call stack:predict_function 报错解决方法2
# tf.config.experimental.set_visible_devices([], 'GPU')
# https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error


from backbone import LeNet,AlexNet,Vgg


# model = LeNet.LeNet5()
# checkpoint_save_path = "./model/LeNet/LeNet5.ckpt"
# model = AlexNet.AlexNet()
# checkpoint_save_path = "./model/AlexNet/AlexNet.ckpt"
model = Vgg.VGG16()
checkpoint_save_path = "./model/Vgg16/Vgg16.ckpt"

model.load_weights(checkpoint_save_path)
# https://tensorflow.google.cn/tutorials/keras/save_and_load?hl=zh-cn
# 虽然文件夹中没有.ckpt文件.
# 没加载文件有时也可以跑通,但是结果随机.

image_path = 'testimg/cat1.jpg' 
# 测试图片中cat1容易错检

img = Image.open(image_path)
img = img.resize((32,32),Image.ANTIALIAS)

img_arr = np.array(img,dtype=float)/255.0

x_predict = img_arr[tf.newaxis,...]
# 升维的原因:https://blog.csdn.net/qq_41660119/article/details/106120433

result = model.predict(x_predict)
# 输出属于各类别概率
pred = tf.argmax(result,axis=1)
# 输出属于最大概率的类别标签,属性为张量

label = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
tf.print(label[int(pred.numpy())])
# 利用.numpy()将张量转化成numpy格式,之后转化为int数据便于检索。