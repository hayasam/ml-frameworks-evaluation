import tensorflow as tf
import os
import argparse
from dataset import Dataset
import learners
import matplotlib.image as mpimg

def get_optimizer(name, learning_rate):
    opt_epsilon = 1.0
    if name == 'adadelta':
        adadelta_rho = 0.95
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=adadelta_rho,
            epsilon=opt_epsilon)
    elif name == 'adagrad':
        adagrad_initial_accumulator_value = 0.1
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=adagrad_initial_accumulator_value)
    elif name == 'adam':
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=opt_epsilon)
    elif name == 'ftrl':
        ftrl_learning_rate_power = -0.5
        ftrl_initial_accumulator_value = 0.1
        ftrl_l1 = 0.0   
        ftrl_l2 = 0.0
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=ftrl_learning_rate_power,
            initial_accumulator_value=ftrl_initial_accumulator_value,
            l1_regularization_strength=ftrl_l1,
            l2_regularization_strength=ftrl_l2)
    elif name == 'momentum':
        momentum = 0.9
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=momentum)
    elif name == 'rmsprop':
        rmsprop_decay = 0.9
        rmsprop_momentum = 0.9
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=rmsprop_decay,
            momentum=rmsprop_momentum,
            epsilon=opt_epsilon)
    elif name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer

def get_activation(name):
    act_fn = tf.nn.relu
    if name == 'elu':
        act_fn = tf.nn.elu
    elif name == 'leaky_relu':
        act_fn = tf.nn.leaky_relu
    elif name == 'relu6':
        act_fn = tf.nn.relu6
    elif name == 'selu':
        act_fn = tf.nn.selu
    elif name == 'tanh':
        act_fn = tf.nn.tanh
    elif name == 'sigmoid':
        act_fn = tf.nn.sigmoid
    return act_fn

def train_lenet(num_epoch, batch_size, weight_decay, keep_p, act_fn, optimizer, checkpoint_path):	
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = Dataset(x_train, y_train.squeeze(), reshape=False, one_hot=True, normalization=True)
    test_data = Dataset(x_test, y_test.squeeze(), reshape=False, one_hot=True, normalization=True)
    X_test, Y_test = test_data.get_data()
    num_sample = train_data.get_num_examples()
    model = learners.LeNet(w_decay=weight_decay, keep_p=keep_p, act_fn=act_fn, optimizer=optimizer)
    saver = tf.train.Saver()
    best_test_accurary = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            for i in range(num_sample // batch_size):
                batch_x, batch_y = train_data.next_batch(batch_size)
                images = sess.run(model.images, feed_dict={model.features: batch_x, model.labels: batch_y})
                mpimg.imsave("./id_{}.png".format(i), images[0].squeeze())
                sess.run(model.train_op, feed_dict={model.features: batch_x, model.labels: batch_y})
                if i % 10 == 0:
                    loss, accurary = sess.run([model.loss, model.accuracy],
                                        feed_dict={model.features: batch_x, model.labels: batch_y})
                    print('[Epoch {}] i: {} Loss: {} Accurary: {}'.format(epoch, i, loss, accurary))
            test_accurary = sess.run(model.accuracy, 
                                    feed_dict={model.features: X_test, model.labels: Y_test})
            print('Test Accurary: {}'.format(test_accurary))
            if best_test_accurary < test_accurary:
                saver.save(sess, checkpoint_path)
                best_test_accurary = test_accurary
            print('Best Test Accurary: {}'.format(best_test_accurary))


def train_DNN(DNN, num_epoch, batch_size, weight_decay, keep_p, optimizer, checkpoint_path):	
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = Dataset(x_train, y_train.squeeze(), reshape=False, one_hot=True, normalization=True)
    test_data = Dataset(x_test, y_test.squeeze(), reshape=False, one_hot=True, normalization=True)
    X_test, Y_test = test_data.get_data()
    num_sample = train_data.get_num_examples()
    model = DNN(w_decay=weight_decay, keep_p=keep_p, optimizer=optimizer)
    saver = tf.train.Saver()
    best_test_accurary = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            for i in range(num_sample // batch_size):
                batch_x, batch_y = train_data.next_batch(batch_size)
                images = sess.run(model.images, feed_dict={model.features: batch_x, model.labels: batch_y})
                mpimg.imsave("./id_{}.png".format(i), images[0].squeeze())
                sess.run(model.train_op, feed_dict={model.features: batch_x, model.labels: batch_y})
                if i % 10 == 0:
                    loss, accurary = sess.run([model.loss, model.accuracy],
                                        feed_dict={model.features: batch_x, model.labels: batch_y})
                    print('[Epoch {}] i: {} Loss: {} Accurary: {}'.format(epoch, i, loss, accurary))
            test_accurary = sess.run(model.accuracy, 
                                    feed_dict={model.features: X_test, model.labels: Y_test})
            print('Test Accurary: {}'.format(test_accurary))
            if best_test_accurary < test_accurary:
                saver.save(sess, checkpoint_path)
                best_test_accurary = test_accurary
            print('Best Test Accurary: {}'.format(best_test_accurary))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='help')
    parser.add_argument('--batch', help='help')
    parser.add_argument('--lr', help='help')
    parser.add_argument('--lambda', help='help')
    parser.add_argument('--keep', help='help')
    args = vars(parser.parse_args())
    num_epoch = int(args['epochs'])
    batch_size = int(args['batch'])
    learning_rate = float(args['lr'])
    weight_decay = float(args['lambda'])
    keep_p = float(args['keep'])
    if not os.path.isdir('./backup'):
        os.mkdir('./backup')
    activations = ['elu']
    optimizers = ['sgd']
    opt_name = 'sgd'
    optimizer = get_optimizer(opt_name, learning_rate)
    model_name = "Inception"
    checkpoint_path = './backup/{}_{}.ckpt'.format(model_name, opt_name)
    train_DNN(learners.Inception, num_epoch, batch_size, weight_decay, keep_p, optimizer, checkpoint_path)
    #for opt_name in optimizers:
    #    optimizer = get_optimizer(opt_name, learning_rate)
    #    for act_name in activations:
    #        act_fn = get_activation(act_name)
    #        checkpoint_path = './backup/lenet_{}_{}.ckpt'.format(opt_name, act_name)
    #        train_lenet(num_epoch, batch_size, learning_rate, weight_decay, keep_p, act_fn, optimizer, checkpoint_path)
    