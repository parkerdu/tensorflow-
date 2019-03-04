"""
3层隐藏层神经网络
隐藏层激活函数：双曲正切
输出层激活函数：softmax
代价函数：softmax交叉熵函数tf.nn.softmax_cross_entropy_with_logits
优化器：adam
过拟合解决：dropout 100%神经元工作

对比4-2 的改变
1.批次:100
2.加入下降的学习率变量
3.降低神经网络维度
"""
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
keep_prob = tf.placeholder(tf.float32)

# 定义一个改变学习率的变量
learning_rate = tf.Variable(0.001, tf.float32)

# 中间层1
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
# 激活函数采用双曲正切
L1 = tf.nn.tanh(tf.matmul(x, W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)  # keep_prob介于0-1之间，设置为0.3
# 就是30%神经元工作

# 中间层2
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)
# # 中间层3
# W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
# b3 = tf.Variable(tf.zeros([1000])+0.1)
# L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
# L3_drop = tf.nn.dropout(L3, keep_prob)
# 输出层
W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W4)+b4)
# 这里W是784*10的矩阵，b是1*10的行向量  总共有6万训练数据，batch_size=100，说明每次过来100行进行训练，所以x是100*784的矩阵
# 但是在算法梯度下降的算法运行时，实际上是按x的一行一行来进行的，每过来一行数据，计算一次梯度，计算一次预测值
# 所以在上面prediction运行时候，底层输入的x实际是1*784  W是784*10  x*W就是1*10 和偏置向b 1*10相加


# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 修改代价函数为softmax交叉熵函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction ))
# 使用梯度下降法
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 是否准确结果存放在一个布尔列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax(y, 1)是返回y向量最大值的下标
# 这里y是一个只有0和1的10维列向量，所以就会返回1所在位置，前面的equal函数是比较两个数是否相等，是返回TURU,

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast将布尔类型转换成浮点型所以true变为1，f变为0，再求平均值就是准确率
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(51):
        sess.run(tf.assign(learning_rate, 0.001*(0.95**epoch)))  # 刚开始需要学习率大点收敛快，迭代到后面趋向收敛应需要更小的学习率 **表示次方
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # batch_xs保存图片数据  batchys保存图片标签
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})  # 每一层70%神经元工作
            # print(sess.run(prediction,feed_dict={x:batch_xs}))
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels, keep_prob:1.0})  # 测试时候还是所有神经元都工作
        # train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print('Period'+str(epoch)+',Test Accuracy'+str(test_acc)+ ',learning rate' + str(sess.run(learning_rate)))



