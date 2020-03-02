from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import mnist_reader

from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.optimizers import SGD

image_height =28
image_width = 28
num_channels = 1
num_classes = 10

def build_model():
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_net(X_train, y_train, X_test, y_test):

    train_data = np.reshape(X_train, (X_train.shape[0], image_height, image_width, num_channels))
    test_data = np.reshape(X_test, (X_test.shape[0],image_height, image_width, num_channels))
    
    from keras.utils import to_categorical
    num_classes = 10
    train_labels_cat = to_categorical(y_train,num_classes)
    test_labels_cat = to_categorical(y_test,num_classes)
    
    print(train_labels_cat.shape)    
    

    # re-scale the image data to values between (0.0,1.0]
    train_data = train_data.astype('float32') / 255.
    test_data = test_data.astype('float32') / 255.

    model = build_model()
    from  keras.callbacks import TensorBoard
    # select rows for train and test
    # trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    Name = 'VGG_benchmark'
    tensorboard = TensorBoard(log_dir="logs/{}".format(Name), profile_batch=100000000)
    
    print(model.summary())
    results = model.fit(train_data, train_labels_cat, 
                    epochs=10, batch_size=64,
                    validation_data=(test_data, test_labels_cat), callbacks=[tensorboard])

    test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=64)
    print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))

def custom_net(X_train, y_train, X_test, y_test):
    
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    # one hot encode target values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    train_norm = X_train.astype('float32')
    test_norm = X_test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model = define_model()
    from  keras.callbacks import TensorBoard
    # select rows for train and test
    # trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    Name = 'test_CNN_dropout30'
    tensorboard = TensorBoard(log_dir="logs/{}".format(Name), profile_batch=100000000)
    history = model.fit(train_norm, y_train, epochs=15, batch_size=64, verbose=1, callbacks=[tensorboard])
    # evaluate model
    _, acc = model.evaluate(test_norm, y_test, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # append scores
    # scores.append(acc)
    # histories.append(history)

def main():
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    
    vgg_net(X_train, y_train, X_test, y_test)
    # custom_net(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
    