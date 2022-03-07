import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from backbone import LeNet,AlexNet,Vgg,InceptionNet,ResNet

# 出现Function call stack:predict_function 报错解决方法
tf.config.experimental.set_visible_devices([], 'GPU')
# https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error
# 或者用如下代码不使用GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = LeNet.LeNet5()
# checkpoint_save_path = "./model/LeNet/LeNet5.ckpt"
# model = AlexNet.AlexNet()
# checkpoint_save_path = "./model/AlexNet/AlexNet.ckpt"
# model = Vgg.VGG16()
# checkpoint_save_path = "./model/Vgg16/Vgg16.ckpt" 
# model = InceptionNet.SimplyInceptionNet(num_block=2,num_classes=10)
# checkpoint_save_path = "./model/SIN/SIN.ckpt"

model = ResNet.SimplyResNet([2,2,2,2])
checkpoint_save_path ="./model/SR/SR.ckpt"

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# model.save_weights('./model/LeNet.h5')
# 用类定义模型,之后调用可能会报错,建议改为函数定义
# https://blog.csdn.net/qq_36758914/article/details/107511743

file = open('./model/weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
