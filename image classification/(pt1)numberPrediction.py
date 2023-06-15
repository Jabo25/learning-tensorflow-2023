# TensorFlow and tf.keras
import tensorflow as tf
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



model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
    # tf.keras.layers.Dense(10)
])
# tf.keras.layers.Dropout(rate=0.1)
# model.add( tf.keras.layers.Dense(4, activation = 'softmax'))




# for i in range(len())
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

historyOfModel  = model.fit(x = x_train, y = y_train, batch_size=20, epochs=30, shuffle=True, validation_split=.2) 

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