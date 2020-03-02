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

def compute_svm(X_train,y_train, X_test, y_test, params):
    print(params[0])
    from sklearn.svm import SVC
    obj = SVC(gamma='scale', kernel=params[0])#, degree=params[1])
    obj.fit(X_train, y_train)
    computed = obj.predict(X_test)
    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_test, computed)
    
    train_computed = obj.predict(X_train)
    train_acc = len(train_computed[train_computed == y_train]) / len(y_train)
    test_acc =len(computed[computed == y_test]) / len(y_test)
    print('Accuracy: ', test_acc) 
    return train_acc, test_acc

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
    results = []
    start = time.time()
    for j in range(2):
        if j ==0:
            X_train, X_test = reduce_dimensionality_LDA()
        if j ==1:
            X_train, X_test, n = reduce_dimensionality_PCA()
        end1 = time.time()
        print('Dimensionality reduced to: ', X_train.shape[1])
        Reduction_time = end1 - start
            
        start = time.time()
        params = ['linear']
        train_accuracy, test_accuracy= compute_svm(X_train, y_train, X_test, y_test, params)
        end = time.time()
        Computation_time = end - start
        results.append([j ,params, train_accuracy*100, test_accuracy*100, Reduction_time, Computation_time]) 
    import tabulate
    import csv
    myFile = open('csvexample_linear.csv', 'w')
    with myFile:
       writer = csv.writer(myFile)
       writer.writerows(results)
       print(tabulate.tabulate(results))

        
if __name__ == "__main__":
    main()



'''
  for i in [0.1, 0.5,1,2,10]:
            start = time.time()
            params = ['poly', i]
            train_accuracy, test_accuracy= compute_svm(X_train, y_train, X_test, y_test, params)
            end = time.time()
            Computation_time = end - start
            results.append([j ,params, train_accuracy*100, test_accuracy*100, Reduction_time, Computation_time]) 
    im
'''
'''
 for j in range(2):
        start = time.time()
        if j ==0:
          X_train, X_test = reduce_dimensionality_LDA()
        if j ==1:
          X_train, X_test, n = reduce_dimensionality_PCA()
        end1 = time.time()
        print('Dimensionality reduced to: ', X_train.shape[1])
        Reduction_time = end1 - start
            
        for i in range(2,9):
            start = time.time()
            params = ['poly', i]
            train_accuracy, test_accuracy= compute_svm(X_train, y_train, X_test, y_test, params)
            end = time.time()
            Computation_time = end - start
            results.append([j ,params, train_accuracy*100, test_accuracy*100, Reduction_time, Computation_time]) 
   '''