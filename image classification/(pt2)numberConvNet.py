


# TensorFlow and tf.keras
import tensorflow as tf
import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop

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





# declair the layers of the network 

# model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear')])# model



# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(rate=0.1),
#     tf.keras.layers.Dense(128, activation = 'tanh'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation = 'softmax')
#     # tf.keras.layers.Dense(10)
# ])

# ?.127/.422/414


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(28, 28, 1))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 14, activation='relu')(img_input)
x = layers.MaxPooling2D(4)(x)
x = layers.Conv2D(32, 7, activation='relu')(img_input)
x = layers.MaxPooling2D(4)(x)
x = layers.Conv2D(64, 4, activation='relu')(img_input)
x = layers.MaxPooling2D(4)(x)

# x = layers.MaxPooling2D(12)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window15
# x = layers.Conv2D(32, 3, activation='relu')(x)
# x = layers.MaxPooling2D(3)(x)

# # Third convolution extracts 64 filters that are 3x3
# # Convolution is followed by max-pooling layer with a 2x2 window
# x = layers.Conv2D(64, 2, activation='relu')(x)
# x = layers.MaxPooling2D(4)(x)

# tf.keras.layers.Dropout(rate=0.1)
# model.add( tf.keras.layers.Dense(4, activation = 'softmax'))


#  Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(rate=0.1)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(10, activation='softmax')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(img_input, output)







# for i in range(len())
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,  optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

historyOfModel  = model.fit(x = x_train, y = y_train, batch_size=20, epochs=10, shuffle=True, validation_split=.2) 

test_loss, test_accuracy = model.evaluate(x_train,  y_train, verbose=2) 


from sklearn.metrics import classification_report


# predictions = model.predict(x_test)
# predicted_label = np.argmax(predictions)
# classification_report = classification_report(x_test, predictions)
# print(classification_report)
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




# model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['MSE' ])


# history = model.fit(np.array(train_features),np.array(train_labels),batch_size=50 , epochs=100 , verbose=1,validation_split = 0.2) 



# MSE = model.evaluate(np.array(test_features), np.array(test_labels), verbose=2)
# print(MSE)






# Do something with the datasets, such as training a model
# ...





# fashion_mnist = tf.keras.datasets.fashion_mnist
# tf.keras.datasets.mnist.load_data(
#     path='mnist.npz'
# )
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Incorrect
# click the backstage view,       click the Options     button, and click the Mail tab. Inside the Outlook Options dialog, you click the Signatures... button. you click the New button. Inside the New Signature dialog, you typed Ken Dishner in the Type a name for this signature input, pressed the Enter key. Inside the Signatures and Stationary dialog in the New messages drop-down, you selected Ken Dishner. Inside the Signatures and Stationary dialog in the Replies//forwards drop-down, you selected Ken Dishner. Inside the Signatures and Stationary dialog, you click the OK button.

# Click the Backstage view. Click the Options button.  In the Outlook Options dialog, click the Mail tab.         Click the Signatures... button.        Click the New button.                                        Type Ken Dishner and click OK.                                                           In the Edit signature box, type Ken Dishner, Head Over Heels Spa. Click the New messages arrow and select Ken Dishner. Click OK in the Signatures and Stationery dialog. Click OK in the Outlook Options dialog.	
# click the backstage view, click the Options ,                                   and click the Mail tab.         click the Signatures... button.        click the New button.                                              you click the OK button.                      you selected Ken Dishner.              Inside the Signatures and Stationary dialog, you click the OK button.