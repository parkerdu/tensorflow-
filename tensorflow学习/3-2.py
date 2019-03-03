import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算总共多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 中间层
a = tf.Variable(np.random.rand([784,25]))
W_1 = 0.3*a - 0.15
b_1 = tf.Variable(tf.zeros([25]))
L_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

# 输出层
W_2 = tf.Variable(tf.zeros([25, 10]))
b_2 = tf.Variable(tf.zeros([10]))
prediction = tf.nn.sigmoid(tf.matmul(L_1, W_2)+b_2)
# 这里W是784*10的矩阵，b是1*10的行向量  总共有6万训练数据，batch_size=100，说明每次过来100行进行训练，所以x是100*784的矩阵
# 但是在算法梯度下降的算法运行时，实际上是按x的一行一行来进行的，每过来一行数据，计算一次梯度，计算一次预测值
# 所以在上面prediction运行时候，底层输入的x实际是1*784  W是784*10  x*W就是1*10 和偏置向b 1*10相加


# 二次代价函数
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 是否准确结果存放在一个布尔列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax(y, 1)是返回y向量最大值的下标
# 这里y是一个只有0和1的10维列向量，所以就会返回1所在位置，前面的equal函数是比较两个数是否相等，是返回TURU,

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast将布尔类型转换成浮点型所以true变为1，f变为0，再求平均值就是准确率
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # batch_xs保存图片数据  batchys保存图片标签
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            # print(sess.run(prediction,feed_dict={x:batch_xs}))
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Period'+str(epoch)+',Test Accuracy'+str(acc))

