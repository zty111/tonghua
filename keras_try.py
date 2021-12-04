from tensorflow.keras.datasets import mnist
from keras.engine import input_spec
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
#to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
#model.add(Dropout(rate=0.25))
model.add(Conv2D(64,(3, 3), padding='same', activation='sigmoid', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64,(3, 3), padding='same', activation='sigmoid'))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(392, activation='sigmoid'))
model.add(Dropout(rate=0.25))
model.add(Dense(196, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20)
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])