import tensorflow as tf
import numpy as np

class ConvolutionalNetwork(object):

    LEARNING_RATE = 0.00000001

    def __init__(self, num_actions, save_path='model_data/agent.ckpt'):
        self._x, self._y, self._action, self._y_hat, self._train_op = ConvolutionalNetwork._build(num_actions)
        self._sess = None
        self.save_path = save_path

    def start_session(self, restore=False):
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self._merged = tf.merge_all_summaries()
        self._writer = tf.train.SummaryWriter("/tmp/mnist_logs", self._sess.graph_def)

        init_op = tf.initialize_all_variables()
        if restore:
            self._saver.restore(self._sess, self.save_path)
        else:
            self._sess.run(init_op)

    def stop_session(self):
        save_path = self._saver.save(self._sess, self.save_path)
        print("Saved to %s" % (save_path,))
        self._sess.close()

    i = 0
    def train_batch(self, batch_x, batch_y, action):
        result = self._sess.run([self._train_op, self._merged],
                                feed_dict={self._x: batch_x, self._y: batch_y, self._action: action})
        summary_str = result[1]
        self._writer.add_summary(summary_str, ConvolutionalNetwork.i)
        self._writer.add_graph(self._sess.graph_def)
        ConvolutionalNetwork.i += 1

    def predict(self, x):
        return self._sess.run(self._y_hat, feed_dict={self._x: x})

    @staticmethod
    def _build(num_actions):
        # Input/output placeholders
        x = tf.placeholder(tf.float32, shape=[None, 84*84*4])
        y = tf.placeholder(tf.float32, shape=[None, num_actions])
        action = tf.placeholder(tf.float32, shape=[None, num_actions])

        # First conv layer
        x_image = tf.reshape(x, [-1,84,84,4])
        W_conv1 = ConvolutionalNetwork.weight_variable([8, 8, 4, 16])
        b_conv1 = ConvolutionalNetwork.bias_variable([16])
        h_conv1 = tf.nn.relu(ConvolutionalNetwork.conv2d(x_image, W_conv1, 4) + b_conv1)
        tf.image_summary("images", x_image, max_images=10, name="input_img")

        # Second conv layer
        W_conv2 = ConvolutionalNetwork.weight_variable([4, 4, 16, 32])
        b_conv2 = ConvolutionalNetwork.bias_variable([32])
        h_conv2 = tf.nn.relu(ConvolutionalNetwork.conv2d(h_conv1, W_conv2, 2) + b_conv2)
#        tf.image_summary("images", h_conv1, max_images=10, name="conv1")
#        tf.image_summary("images", h_conv2, max_images=10, name="conv2")

        # Fully connected hidden layer
        h_conv2_flat = tf.reshape(h_conv2, [-1, 3872])
        W_fc1 = ConvolutionalNetwork.weight_variable([3872, 256])
        b_fc1 = ConvolutionalNetwork.bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # Output layer
        W_fc2 = ConvolutionalNetwork.weight_variable([256, num_actions])
        b_fc2 = ConvolutionalNetwork.bias_variable([num_actions])
        y_hat = tf.matmul(h_fc1, W_fc2) + b_fc2

        y_hat_diff = tf.reduce_sum(tf.mul(y_hat, action), reduction_indices=1)
        y_sum = tf.reduce_sum(y, reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y_sum - y_hat_diff))
        tf.scalar_summary(cost.op.name, cost)
        train_op = tf.train.GradientDescentOptimizer(ConvolutionalNetwork.LEARNING_RATE).minimize(cost)

        return (x, y, action, y_hat, train_op)

    @staticmethod
    def weight_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride=1):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
