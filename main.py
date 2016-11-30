from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_train, height, width = X_train.shape
n_test, _, _ = X_test.shape

# print(n_train, n_test, height, width)

X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
X_test = X_test.reshape(n_test, 1, height, width).astype('float32')

# normalize
X_train /= 255
X_test /= 255

n_classes = 10

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

model = Sequential()

# number of convolutional filters
n_filters = 32

# convolution filter size (n_conv x n_conv filter)
n_conv = 3

# pooling window size (n_pool x n_pool window)
n_pool = 2

# first convolutional layer
model.add(
    Convolution2D(
        n_filters, n_conv, n_conv,
        border_mode='valid',    # narrow convolution, no spill over at border
        input_shape=(1, height, width),
    )
)
model.add(Activation('relu'))

# second convolutional layer
model.add(Convolution2D(n_filters, n_conv, n_conv))
model.add(Activation('relu'))

# pooling layer
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))

# flatten data for 1D layers
model.add(Flatten())

# Dense(n_output) fully connected hidden layer
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# number of examples to look at during each training session
batch_size = 128

# number of times to run through full sets of examples
n_epochs = 10

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    nb_epoch=n_epochs,
    validation_data=(X_test, y_test)
)

# to see results
loss, accuracy = model.evaluate(X_test, y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)
