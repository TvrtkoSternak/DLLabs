import tensorflow as tf
from Lab1.tf_deep import TF_deep
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import Lab0.data as data

tf.app.flags.DEFINE_string('data_dir',
  '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(
  tf.app.flags.FLAGS.data_dir, one_hot=True)

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

tf_deep_1 = TF_deep([784, 10])
tf_deep_2 = TF_deep([784, 100, 10])
tf_deep_3 = TF_deep([784, 100, 100, 10], param_delta = 0.001)
tf_deep_4 = TF_deep([784, 100, 100, 100, 10], param_delta = 0.001)

models = list()
models.append(tf_deep_1)
# models.append(tf_deep_2)
# models.append(tf_deep_3)
# models.append(tf_deep_4)

def prepare_models(mode="train", num_epochs = 100):
        if mode == "train":
                print("training models")
                tf_deep_1.train(mnist.train.images, mnist.train.labels, num_epochs)
                tf_deep_1.save_model("/tmp/model1.ckpt")
                # tf_deep_2.train(mnist.train.images, mnist.train.labels, num_epochs)
                # tf_deep_2.save_model("/tmp/model2.ckpt")
                # tf_deep_3.train(mnist.train.images, mnist.train.labels, num_epochs)
                # tf_deep_3.save_model("/tmp/model3.ckpt")
                # tf_deep_4.train(mnist.train.images, mnist.train.labels, num_epochs)
                # tf_deep_4.save_model("/tmp/model4.ckpt")
        elif mode == "load":
                print("loading models")
                tf_deep_1.load_model("/tmp/model1.ckpt")
                tf_deep_2.load_model("/tmp/model2.ckpt")
                tf_deep_3.load_model("/tmp/model3.ckpt")
                tf_deep_4.load_model("/tmp/model4.ckpt")
                print("models loaded")

def test_models():
        for i, model in enumerate(models):
                print("model{0} results:".format(i+1))
                # dohvati vjerojatnosti na skupu za učenje
                probs = model.eval(mnist.train.images)

                # ispiši performansu (preciznost i odziv po razredima)
                Y = model.eval_class(mnist.test.images)

                accuracy, pr, M = data.eval_perf_multi(Y, mnist.test.labels.argmax(axis=1))
                print("Accuracy: ", accuracy)
                print("Precision / Recall: ", pr)
                print("Confusion Matrix: ", M)


def terminate_models():
        for model in models:
                model.session.close()

if __name__ == "__main__":
        # inicijaliziraj generatore slučajnih brojeva
        np.random.seed(100)
        tf.set_random_seed(100)

        prepare_models(mode = "train", num_epochs=1000)
        test_models()
        terminate_models()