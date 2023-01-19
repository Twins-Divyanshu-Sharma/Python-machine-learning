import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train /= 255
X_test /= 255

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)) )
model.add(MaxPooling2D((2, 2)) )
model.add(Flatten())
model.add(Dense(50, activation='softmax'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test,y_test))

predictions = model.evaluate(X_test, Y_test)
print(predictions)

