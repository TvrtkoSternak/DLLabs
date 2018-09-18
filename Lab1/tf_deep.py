import tensorflow as tf
import numpy as np
import Lab0.data as data
import matplotlib.pyplot as plt


# creation of weights and biases

def weights_and_biases(a, b):
    w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
    b = tf.Variable(tf.zeros([b]))
    return w, b

class TF_deep:

    def __init__(self, config, param_delta = 0.01, param_lambda = 0.0001):
        self.X = tf.placeholder(tf.float32, [None, config[0]])
        self.Yoh_ = tf.placeholder(tf.float32, [None, config[-1]])
        self.logits = self.X
        self.L2_measure = 0

        for i in range(config.__len__() - 1):
            W_i, b_i = weights_and_biases(config[i], config[i+1])
            self.L2_measure += tf.nn.l2_loss(W_i)
            if i != config.__len__() - 2:
                self.logits = tf.nn.tanh(tf.matmul(self.logits, W_i) + b_i)
            else:
                self.logits = tf.matmul(self.logits, W_i) + b_i

        self.probs = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                    logits = self.logits, labels = self.Yoh_))
        self.loss = tf.reduce_mean(self.loss + param_lambda * self.L2_measure)

        self.saver = tf.train.Saver()

        self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)
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
            _, loss = self.session.run([self.train_step, self.loss], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if epoch%100 == 0:
                print("epoch {0} loss = {1}".format(epoch, loss))

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

    def save_model(self, save_path):
        self.saver.save(self.session, save_path=save_path)

    def load_model(self, save_path):
        self.saver.restore(self.session, save_path=save_path)

if __name__ == "__main__":
        # inicijaliziraj generatore slučajnih brojeva
        np.random.seed(100)
        tf.set_random_seed(100)

        # instanciraj podatke X i labele Yoh_

        X, Y_ = data.sample_gmm_2d(6, 2, 10)

        Yoh_ = data.class_to_onehot(Y_)

        # izgradi graf:
        tf_deep = TF_deep([2, 10, 10, 2])

        # nauči parametre:
        tf_deep.train(X, Yoh_, 10000)

        # dohvati vjerojatnosti na skupu za učenje
        probs = tf_deep.eval(X)
        print(probs)

        # ispiši performansu (preciznost i odziv po razredima)
        Y = tf_deep.eval_class(X)
        print(Y)
        accuracy, pr, M = data.eval_perf_multi(tf_deep.eval_class(X), Y_)
        print("Accuracy: ", accuracy)
        print("Precision / Recall: ", pr)
        print("Confusion Matrix: ", M)

        # iscrtaj rezultate, decizijsku plohu
        rect = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(tf_deep.eval_probs, rect, offset=0.5)
        data.graph_data(X, Y_, Y_)
        plt.show()