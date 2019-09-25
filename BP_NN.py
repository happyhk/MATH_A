# 引入相关库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 读取数据
train_data = np.array(pd.read_csv("./train_data/traindatasets.csv"))
test_data = np.array(pd.read_csv("./last_train_data/test_112501.csv"))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 提取特征列,即X (共7列，代表7个变量)
train_feature = np.array(train_data[:, [0, 1, 2, 3, 4, 5, 6]])
# 提取预测结果列，即Y
train_label = np.array(train_data[:, [7]])
# 提取测试集特征列
test_xs = np.array(test_data[:, [0, 1, 2, 3, 4, 5, 6]])
print(test_data.shape)
print(train_feature.shape)
print(train_label.shape)
# 搭建神经网络
# 定义x y
model_input = tf.placeholder(tf.float32, [None, 7])  # 长度为7，代表7个特征
y = tf.placeholder(tf.float32, [None, 1])  # 长度为1，代表要预测的变量只有1个
# train_feature = preprocessing.scale(train_feature)  # 数据预处理，归一化
# test_xs = preprocessing.scale(test_x)  # 也对测试集进行预处理
print(test_xs.shape)

# 定义神经网络隐藏层

# 初始化权值。 为18*20矩阵  20代表20个神经元
Weights_L1 = tf.Variable(tf.random_normal([7, 20]))
# 偏置矩阵
biases_L1 = tf.Variable(tf.zeros([1, 20]))
Wx_plus_b_L1 = tf.matmul(model_input, Weights_L1) + biases_L1
# 激活函数私有tanh
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([20, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 20]))
model_output = tf.matmul(L1, Weights_L2) + biases_L2

# 代价函数
loss = tf.reduce_mean(tf.square(y - model_output))


# 定义优化器。使用动量法 也可以使用随机梯度下降法等
train_step = tf.train.MomentumOptimizer(0.05, 0.05).minimize(loss)

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

with tf.Session() as sess:
    # 初始化变量

    sess.run(tf.global_variables_initializer())
    # writer=tf.summary.FileWriter("gra",graph=tf.get_default_graph())
    print(sess.run(loss, feed_dict={model_input: train_feature, y: train_label}))
    for i in range(200):
        sess.run(train_step, feed_dict={model_input: train_feature, y: train_label})
        print("epoch:", i)
        # print(sess.run(L1,feed_dict={x: train_feature, y: train_label}))
        print(sess.run(loss, feed_dict={model_input: train_feature, y: train_label}))

    prd = sess.run(model_output, feed_dict={model_input: test_xs})  # 获取对测试集的预测结果
    print(prd)
    print(type(prd))
    tf.saved_model.simple_save(sess, "./model/", inputs={"myInput": model_input}, outputs={"myOutput": model_output})
