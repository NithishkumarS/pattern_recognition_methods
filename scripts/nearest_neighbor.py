import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import mnist_reader
from bayes_classifier import segregate_data
from PCA import *

X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
covariance = [0.99, .95, .9, .8]
storage = []

X_train, X_test = reduce_dimensionality_LDA()
kVals = range(1, 20, 2)
accuracies = []

for k in range(1, 20, 2):
    print(k)
    start = time.time()
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    end = time.time()
    total_time = end - start
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    print('tiem:', total_time)
    storage.append([k, total_time,score*100 ])

import tabulate
print(tabulate.tabulate(storage))