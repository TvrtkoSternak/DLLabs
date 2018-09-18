import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time

LOSS_SCALE = 0.001
LEARNING_RATE = 0.001
img_height = 32
img_width = 32
num_channels = 3
DATA_DIR = '/home/tvrtko/Fer/DubokoUcenje/Labosi/lab2/cifar-10-batches-py'
PLOT_SAVE_DIR = '/home/tvrtko/Fer/DubokoUcenje/Labosi/lab2/plots'
VISUALISATION_SAVE_DIR = ''
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, img_height, img_width, num_channels])

    # regularisers
    l2_reg1 = tf.contrib.layers.l2_regularizer(
                                scale=LOSS_SCALE
                            )

    l2_reg2 = tf.contrib.layers.l2_regularizer(
                                scale=LOSS_SCALE
                            )

    l2_reg3 = tf.contrib.layers.l2_regularizer(
                                scale=LOSS_SCALE
                            )

    l2_reg4 = tf.contrib.layers.l2_regularizer(
                                scale=LOSS_SCALE
                            )


    # convolution1

    conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg1
                )

    # maxpooling1

    pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=3,
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
                pool_size=3,
                strides=2
                )

    # flatten

    flat = tf.layers.flatten(
                inputs=pool1
                )

    # fc1

    fc1 = tf.layers.dense(
                inputs=flat,
                units=256,
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg3
                )

    # fc2

    fc2 = tf.layers.dense(
                inputs=fc1,
                units=128,
                activation=tf.nn.relu,
                kernel_regularizer=l2_reg4
                )

    # logits

    logits = tf.layers.dense(
                inputs=fc2,
                units=10
                )

    # loss calculation

    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    total_loss = loss + l2_loss

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # train model
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(
                            loss=total_loss,
                            global_step=tf.train.get_global_step()
                            )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # return evaluation
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)

def evaluate(logits, loss, train_x, train_y):
    pass

def main():
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
      subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
      train_x = np.vstack((train_x, subset['data']))
      train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0,1,2))
    data_std = train_x.std((0,1,2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    # Create the Estimator
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model4")

    # Set up logging for predictions
    tf.logging.set_verbosity(tf.logging.INFO)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=100,
        num_epochs=1,
        shuffle=True)

    eval_input_fn_valid = tf.estimator.inputs.numpy_input_fn(
        x={"x": valid_x},
        y=valid_y,
        num_epochs=1,
        shuffle=False)

    eval_input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        num_epochs=1,
        shuffle=False)

    for epoch in range(5):
        cifar_classifier.train(
            input_fn=train_input_fn,
        )

        eval_results_train = cifar_classifier.evaluate(input_fn=eval_input_fn_train)
        eval_results_valid = cifar_classifier.evaluate(input_fn=eval_input_fn_valid)

        test_shit = 1

        train_loss = eval_results_train["loss"]
        valid_loss = eval_results_valid["loss"]
        train_acc = eval_results_train["accuracy"]
        valid_acc = eval_results_valid["accuracy"]

        print(eval_results_train)
        print(train_loss)
        print(train_acc)
        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [valid_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [valid_acc]
        plot_data['lr'] += [optimizer._lr]

    plot_training_progress(PLOT_SAVE_DIR, plot_data)

if __name__ == "__main__":
    main()
