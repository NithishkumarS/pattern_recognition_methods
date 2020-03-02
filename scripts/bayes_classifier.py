import numpy as np
import mnist_reader
from PCA import reduce_dimensionality_LDA, reduce_dimensionality_PCA
import time

def segregate_data(X_train, y_train):
    class_list = []
    for label in range(10):
        idx = np.where(y_train == label)
        elements = X_train[idx]
        class_list.append(elements)
    return class_list

def compute_class_statistics(X_train, y_train):
    stats = {}
    class_list = segregate_data(X_train, y_train)
    obj = MLE(X_train, y_train)
    for i in range(len(class_list)):
        stats[i] = [i, obj.compute_mean(class_list[i]), obj.compute_var(class_list[i]) ]
    return stats

def test(X_train, y_train, X_test, y_test):
    stats = compute_class_statistics(X_train, y_train)
    storage = []
    for i in stats.keys():
            idx, mu, cov = stats[i]
            mu = mu.reshape(len(mu),1)
            a = (np.subtract(X_test.T, mu)).T
            temp = (-.5)*np.sum(np.multiply(np.dot(a, np.linalg.pinv(cov)),a), axis=1)
            storage.append(temp)
    print('ytest: ', y_test)
    idx = np.argmax(storage,axis=0)
    
    print('Test data: ',len(y_test))
    print('computed correct: ', len(np.where(np.array(idx) == y_test)[0]))
    return len(np.where(np.array(idx) == y_test)[0]) / len(y_test)

class MLE:
    def __init__(self,X_train, y_train):
        self.x = X_train
        self.y = y_train
    def return_mean(self):
        return self.mean
    def return_variance(self):
        return self.var
    def compute_mean(self, x):
        return np.mean(x, axis=0)
    def compute_var(self,x):
        return np.cov(x,rowvar=False)
      
def main():
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    start = time.time()
    covariance = [0.99, .95, .9, .8]
    X_train, X_test = reduce_dimensionality_LDA()
#         X_train, X_test, n = reduce_dimensionality_PCA(var)
    print('Dimensionality reduced to: ', X_train.shape[1])
    accuracy= test(X_train, y_train, X_test, y_test)
    end = time.time()
    total_time = end - start
    print('time: ', total_time)
    
if __name__ == "__main__":
    main()