import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(50, activation='softmax'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test,y_test))

predictions = model.evaluate(X_test, Y_test)
print(predictions)


