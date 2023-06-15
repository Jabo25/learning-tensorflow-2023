# this is the 3rd itteration of the model which uses data augmetnation ot imporve the accurazy of to 3% or even low as 2.8 on the validation set. 
import tensorflow as tf

# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


print(tf.__version__)


import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0



class_names = [0,1,2,3,4,5,6,7,8,9,10]

# Print the shape of the datasets
print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)


# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(28, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(num_classes)
#     ])
# model = dataAugmentation

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(28, (7, 7), activation='relu', input_shape = (28,28,1)),
    # tf.keras.layers.RandomZoom(0.1,fill_mode='reflect',interpolation='bilinear',seed=None,fill_value=1),
    # tf.keras.layers.RandomCrop(0.2,fill_mode='reflect',interpolation='bilinear',seed=None,fill_value=1),
    # the previous record was with the .1,.1 zoom
    tf.keras.layers.RandomTranslation(0.1,0.1,fill_mode='reflect',interpolation='bilinear',seed=None,fill_value=1),
    # tf.keras.layers.RandomFlip(mode="horizontal", seed=None),
    # tf.keras.layers.RandomRotation(.2,fill_mode='reflect',interpolation='bilinear',seed=None,fill_value=1.0),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# model.build((None,) + (28, 28, 1)) 

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# tf.keras.layers.Dropout(rate=0.1)
# model.add( tf.keras.layers.Dense(4, activation = 'softmax'))





# for i in range(len())
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
historyOfModel  = model.fit(x = x_train, y = y_train, batch_size=150, epochs=17, shuffle=True, validation_split=.2) 
test_loss, test_accuracy = model.evaluate(x_train,  y_train, verbose=2) 





from sklearn.metrics import classification_report
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
classification_report = classification_report(y_test, predicted_labels)
print(classification_report)



plt.figure(figsize=(10,10))
errorTotal = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()
print("")

print("printing out the errors")

plt.figure(figsize=(10,10))


errorTotal = 0
plt.figure(figsize=(10, 10))  # Adjust the figure size if needed

for i in range(len(y_test)):
    if predicted_labels[i] == y_test[i]:
        errorTotal = errorTotal
    else:
        errorTotal = errorTotal + 1
        plt.subplot(5, 5, errorTotal)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(predicted_labels[i])

    if errorTotal >= 25:
        break

plt.tight_layout()  # Optional, to improve the layout of subplots
plt.show()
print("")

