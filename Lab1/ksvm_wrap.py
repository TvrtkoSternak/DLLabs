from sklearn import svm
import scipy
import numpy as np
import Lab0.data as data
import matplotlib.pyplot as plt

'''Metode:
  __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
    Konstruira omotač i uči RBF SVM klasifikator
    X,Y_:            podatci i točni indeksi razreda
    param_svm_c:     relativni značaj podatkovne cijene
    param_svm_gamma: širina RBF jezgre

  predict(self, X)
    Predviđa i vraća indekse razreda podataka X

  get_scores(self, X):
    Vraća klasifikacijske mjere podataka X

  suport
    Indeksi podataka koji su odabrani za potporne vektore
'''

class ksvm_wrap:
    def __init__(self, X, Y_, param_svm_c=1.0, param_svm_gamma='auto'):
        self.clf = svm.SVC(C = param_svm_c , gamma = param_svm_gamma)
        self.clf.fit(X, Y_)
        self.support = list()

        for vector in self.clf.support_vectors_:
            self.support.append(np.where((X == vector).all(axis=1)))

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X, Y_):
        accuracy, pr, M = data.eval_perf_multi(self.predict(X), Y_)
        print("Accuracy: ", accuracy)
        print("Precision / Recall: ", pr)
        print("Confusion Matrix: ", M)

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    clf_classifier = ksvm_wrap(X, Y_)

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(clf_classifier.predict, rect, offset=0.5)
    data.graph_data(X, Y_, clf_classifier.predict(X), clf_classifier.support)
    clf_classifier.get_scores(X, Y_)
    plt.show()