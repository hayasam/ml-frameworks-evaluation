import tensorflow as tf
import models
from nets import inception
from nets import resnet_v1
slim = tf.contrib.slim

class LeNet:

    def __init__(self, w_decay, keep_p, act_fn, optimizer, input_shape=[None, 28, 28], input_reshape=[-1, 28, 28, 1]):
        self.features = tf.placeholder(tf.float32, input_shape)
        self.images = tf.reshape(self.features, input_reshape)
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.labels, tf.int32)
        
        with tf.variable_scope("LeNet") as scope:
            self.train_digits = models.lenet(images=self.images, dropout_keep_prob=keep_p, weight_decay=w_decay, is_training=True, act_fn=act_fn)
            scope.reuse_variables()
            self.pred_digits = models.lenet(images=self.images, dropout_keep_prob=keep_p, weight_decay=w_decay, is_training=False, act_fn=act_fn)
        self.logits = self.pred_digits
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.train_digits))
        self.train_op = optimizer.minimize(self.loss)


class Inception():

    def __init__(self, w_decay, keep_p, optimizer, input_shape=[None, 28, 28], input_reshape=[-1, 28, 28, 1]):
        self.img_size = inception.inception_v1.default_image_size
        self.features = tf.placeholder(tf.float32, input_shape)
        self.images = tf.reshape(self.features, input_reshape)
        self.images = tf.image.resize_images(self.images, [self.img_size, self.img_size])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.labels, tf.int32)

        with tf.variable_scope("Inception") as scope:
            self.train_digits, _ = inception.inception_v1(self.images, num_classes=10, is_training=True)
            scope.reuse_variables()
            self.pred_digits, _ = inception.inception_v1(self.images, num_classes=10, is_training=False)
        
        self.logits = self.pred_digits
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.train_digits))
        self.train_op = optimizer.minimize(self.loss)
        

