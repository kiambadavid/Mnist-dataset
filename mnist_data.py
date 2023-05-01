import keras
from matplotlib import pyplot as plt
from keras import layers
from keras.datasets import mnist

# load data
mnist_data = keras.datasets.mnist
print(mnist)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# explore the data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# process the data by normalizing the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape data to 2D array
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# build the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_train)
print('Test Accuracy:', test_acc)

# make predictions
predictions = model.predict(x_test[:5])
print(predictions)

# fine-tune the model
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
print(history)

# visualize the model
# metric = accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# visualize the model
# loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# visualize the model
# loss againist accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('accuracy')
plt.ylabel('loss')
plt.legend()
plt.show()
