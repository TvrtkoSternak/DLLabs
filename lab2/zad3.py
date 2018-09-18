import tensorflow as tf
import numpy as np
import logging as _logging
import skimage as ski
import os
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
import skimage.io


LOSS_SCALE = 0.001
DATA_DIR = '/home/tvrtko/Fer/DubokoUcenje/Labosi/MNIST/'
SAVE_DIR = "/home/tvrtko/Fer/DubokoUcenje/Labosi/lab2/out_zad3/"

def cnn_model_fn(X, Y_, weight_decay = LOSS_SCALE):
    # Input Layer
    input_layer = tf.reshape(X, [-1, 28, 28, 1])

    # regularisers
    l2_reg1 = tf.contrib.layers.l2_regularizer(
                                scale=weight_decay,
                            )

    l2_reg2 = tf.contrib.layers.l2_regularizer(
                                scale=weight_decay,
                            )

    l2_reg3 = tf.contrib.layers.l2_regularizer(
                                scale=weight_decay,
                            )


    # convolution1

    conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg1,
                name = 'conv1_1'
                )

    # maxpooling1

    pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=2,
                strides=2
                )

    # convolution2

    conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg2
                )

    # maxpooling2

    pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=2,
                strides=2
                )

    # flatten

    flat = tf.layers.flatten(
                inputs=pool2
                )

    # fc1

    fc1 = tf.layers.dense(
                inputs=flat,
                units=512,
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg3
                )

    # logits

    logits = tf.layers.dense(
                inputs=fc1,
                units=10
                )

    # loss calculation

    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=logits)
    total_loss = loss + l2_loss

    return logits, total_loss, conv1

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[3]
  num_channels = w.shape[2]
  k = w.shape[0]
  assert w.shape[0] == w.shape[1]
  w = w.reshape(k, k, num_channels, num_filters)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, 3])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)



def train(sess, train_x, train_y, valid_x, valid_y, kernel, lr_policy, logits, total_loss, optimiser, X, Y_, lr):
  lr_policy = lr_policy
  batch_size = 250
  max_epochs = 8
  save_dir = SAVE_DIR
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  for epoch in range(1, max_epochs+1):
    if epoch in lr_policy:
      solver_config = lr_policy[epoch]
    cnt_correct = 0
    #for i in range(num_batches):
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    #for i in range(100):
    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]

      logits_val, total_loss_val, _ = sess.run([logits, total_loss, optimiser], feed_dict={X:batch_x, Y_:batch_y, lr:solver_config['lr']})
      #logits = forward_pass(net, batch_x)
      #loss_val = loss.forward(logits, batch_y)
      # compute classification accuracy
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      #grads = backward_pass(net, loss, logits, batch_y)
      #sgd_update_params(grads, solver_config)

      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, total_loss_val[1]))
      if i % 100 == 0:
        conv1_var = tf.contrib.framework.get_variables('conv1_1')[0]
        conv1_weights = conv1_var.eval(session=sess)
        draw_conv_filters(epoch, i, conv1_weights, SAVE_DIR)
        #draw_conv_filters(epoch, i*batch_size, net[3])
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate(sess, "Validation", valid_x, valid_y, logits, total_loss, X, Y_)

def evaluate(sess, name, x, y, logits, total_loss, X, Y_):
  print("\nRunning evaluation: ", name)
  batch_size = 250
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val, loss_val = sess.run([logits, total_loss], feed_dict={X:batch_x, Y_:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    loss_avg += loss_val
    #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg[1])


def main():
    tf.reset_default_graph()
    # Load training and eval data
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
    Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    logits, total_loss, conv1 = cnn_model_fn(X, Y_)
    lr = tf.placeholder(tf.float32)
    optimiser = tf.train.GradientDescentOptimizer(lr).minimize(total_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    lr_policy = {1: {'lr': 1e-4}, 3: {'lr': 1e-5}, 5: {'lr': 1e-6}, 7: {'lr': 1e-7}}

    train(sess, train_x, train_y, valid_x, valid_y, conv1, lr_policy, logits, total_loss, optimiser, X, Y_, lr)
    evaluate(sess, "test", test_x, test_y, logits, total_loss, X, Y_)

if __name__ == "__main__":
    main()



