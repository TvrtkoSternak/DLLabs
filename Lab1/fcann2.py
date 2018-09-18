import Lab0.data as data
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.vstack(e_x.sum(axis=1))


def fcnn2_decfun_probs(X, W1, b1, W2, b2):
    def classify(X):
        return fcann2_classify(X, W1, b1, W2, b2)[:, 1]

    return classify


def fcnn2_decfun_class(X, W1, b1, W2, b2):
    def classify(X):
        return fcann2_classify(X, W1, b1, W2, b2).argmax(axis=1)

    return classify


def fcann2_train(X, Y, num_hidden_neurons, param_n_iter, param_delta, param_lambda, print_diagnostics_step = 10):
    num_examples = X.shape[0]
    num_classes = np.max(Y) + 1
    W1 = np.random.randn(X.shape[1], num_hidden_neurons)
    b1 = np.zeros((1, num_hidden_neurons))
    W2 = np.random.randn(num_hidden_neurons, num_classes)
    b2 = np.zeros((1, num_classes))
    param_n_iter = param_n_iter
    param_delta = param_delta
    param_lambda = param_lambda

    for i in range(param_n_iter):
        # forward pass
        s1 = np.dot(X, W1) + b1
        h1 = relu(s1)
        s2 = np.dot(h1, W2) + b2

        # calculate class probabilities
        probs = softmax(s2)

        if print_diagnostics_step >0 and i % print_diagnostics_step == 0:
            # calculate loss
            log_class_probs = np.log(probs[range(num_examples), Y])
            loss = -np.sum(log_class_probs) / num_examples
            print("iteration {0} loss = {1}".format(i, loss))

        # derivation of loss
        dL = probs
        dL[range(num_examples), Y] -= 1
        dL /= num_examples

        grad_W2 = np.dot(h1.T, dL)
        grad_b2 = np.sum(dL, axis=0)

        dL_1 = np.dot(dL, W2.T)
        dL_relu = dL_1
        dL_relu[h1 <= 0] = 0

        grad_W1 = np.dot(X.T, dL_relu)
        grad_b1 = np.sum(dL_relu, axis=0)

        # regularisation
        grad_W2 += param_lambda * W2
        grad_W1 += param_lambda * W1

        # weights update

        W2 += -param_delta * grad_W2
        W1 += -param_delta * grad_W1
        b2 += -param_delta * grad_b2
        b1 += -param_delta * grad_b1

        pass

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    s1 = np.dot(X, W1) + b1
    h1 = relu(s1)
    s2 = np.dot(h1, W2) + b2
    return softmax(s2)


if __name__ == "__main__":
    # init random seeds
    np.random.seed(100)

    # create data
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # train model
    W1, b1, W2, b2 = fcann2_train(X, Y_, 5, 10000, 0.05, 1e-3, print_diagnostics_step=1000)

    # print diagnostics
    Y = fcnn2_decfun_class(X, W1, b1, W2, b2)(X)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy)
    print("Precision / Recall: ", pr)
    print("Confusion Matrix: ", M)

    # graph the decision surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    decision_function = fcnn2_decfun_probs(X, W1, b1, W2, b2)
    data.graph_surface(decision_function, rect, offset=0.5)
    data.graph_data(X, Y_, fcnn2_decfun_class(X, W1, b1, W2, b2))
    plt.show()
