import tensorflow as tf
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

# 创建一个简单的神经网络 只有输入输出层
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



prediction = tf.nn.softmax(tf.matmul(x, W)+b)
# 这里W是784*10的矩阵，b是1*10的行向量  总共有6万训练数据，batch_size=100，说明每次过来100行进行训练，所以x是100*784的矩阵
# 但是在算法梯度下降的算法运行时，实际上是按x的一行一行来进行的，每过来一行数据，计算一次梯度，计算一次预测值
# 所以在上面prediction运行时候，底层输入的x实际是1*784  W是784*10  x*W就是1*10 和偏置向b 1*10相加


# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 修改代价函数为softmax交叉熵函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction ))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

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

