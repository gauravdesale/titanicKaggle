
import tensorflow as tf
import input_data

sess = tf.InteractiveSession()
titanicData = input_data.read_data_sets("..Downloads/train.csv", one_hot=True)


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
num_examples= 33500

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 20 # MNIST data input (img shape: 28*28)
n_classes = 4 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("int", [None, n_input])
y_ = tf.placeholder("int", [None, n_classes])

W = tf.Variable(tf.zeros([x.shape, y.shape]))
b = tf.Variable(tf.zeros(y.shape))
sess.run(tf.global_variables_initializer)


with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
