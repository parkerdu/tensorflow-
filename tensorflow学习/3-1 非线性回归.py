import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
# x_data = np.random.rand(200)  # 任意的200个点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # -0.5到0.5之间的相同间隔200个点
print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])# [None, 1]定义这个占位符的shape位任意行，单列
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网洛中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  # 定义一个1行10列的中间层权重
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  #
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 中间层的输出 激活函数为双曲正切函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)  # scatter 画散点图
    plt.plot(x_data,prediction_value, 'r-', lw=5)  #红色实线
    plt.show()