import numpy as np
import tensorflow as tf
import Lab0.data as data
import matplotlib.pyplot as plt


class TFLogreg:
  def __init__(self, D, C, param_delta=0.5, param_lambda = 0.2):
    """Arguments:
       - D: dimensions of each datapoint
       - C: number of classes
       - param_delta: training step
    """

    # definicija podataka i parametara:
    # definirati self.X, self.Yoh_, self.W, self.b
    # ...

    self.X = tf.placeholder(tf.float32, [None, D])
    self.Yoh_ = tf.placeholder(tf.float32, [None, C])
    self.W = tf.Variable(tf.zeros([D, C]))
    self.b = tf.Variable(tf.zeros([C]))
    # formulacija modela: izračunati self.probs
    #   koristiti: tf.matmul, tf.nn.softmax
    # ...

    self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

    # formulacija gubitka: self.loss
    #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
    # ...

    self.unreg_loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs), reduction_indices=1))
    self.loss = self.unreg_loss + param_lambda * tf.nn.l2_loss(self.W)

    # formulacija operacije učenja: self.train_step
    #   koristiti: tf.train.GradientDescentOptimizer,
    #              tf.train.GradientDescentOptimizer.minimize
    # ...

    self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

    # instanciranje izvedbenog konteksta: self.session
    #   koristiti: tf.Session
    # ...

    self.session = tf.Session()

  def train(self, X, Yoh_, param_niter):
    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    # incijalizacija parametara
    #   koristiti: tf.initialize_all_variables
    # ...

    init = tf.global_variables_initializer()

    # optimizacijska petlja
    #   koristiti: tf.Session.run
    # ...

    self.session.run(init)

    for epoch in range(param_niter):
        self.session.run([self.train_step, self.loss], feed_dict={self.X: X,self.Yoh_: Yoh_})

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    #   koristiti: tf.Session.run
    probs = self.session.run(self.probs, feed_dict={self.X: X})
    return probs

  def eval_probs(self, X):
    probs = self.session.run(self.probs, feed_dict={self.X: X})
    return probs[:, 1]

  def eval_class(self, X):
    probs = self.session.run(self.probs, feed_dict={self.X: X})
    return probs.argmax(axis=1)


if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)
  tf.set_random_seed(100)

  # instanciraj podatke X i labele Yoh_

  X, Y_ = data.sample_gauss(3, 100)

  Yoh_ = data.class_to_onehot(Y_)

  # izgradi graf:
  tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.1)

  # nauči parametre:
  tflr.train(X, Yoh_, 1000)

  # dohvati vjerojatnosti na skupu za učenje
  probs = tflr.eval(X)

  # ispiši performansu (preciznost i odziv po razredima)
  Y = tflr.eval_class(X)
  print(Y)
  accuracy, pr, M = data.eval_perf_multi(tflr.eval_class(X), Y_)
  print("Accuracy: ", accuracy)
  print("Precision / Recall: ", pr)
  print("Confusion Matrix: ", M)

  # iscrtaj rezultate, decizijsku plohu
  rect = (np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(tflr.eval_class, rect, offset=1)
  data.graph_data(X, Y_, Y_)
  plt.show()
