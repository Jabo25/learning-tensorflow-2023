import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_data_flow_ops import fifo_queue
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras import backend as K

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)



#create dataframe
data = pd.read_csv('classification/car.csv')
print(data.head())


#transformas all fo the buying values into a list of apropriate int encoded values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
Class = le.fit_transform(list(data["class"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))



predict = "class"
features = list(zip(buying,maint,doors, persons,lug_boot,safety))
label = list(Class)

train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, label, test_size = 0.1)


class_labels = np.unique(train_labels)
class_weights = compute_sample_weight(class_weight='balanced', y=label)


#declair the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='tanh'))
tf.keras.layers.Dropout(rate=0.1)
model.add( tf.keras.layers.Dense(4, activation = 'softmax'))


# Compile your model using the custom precision metric
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

historyOfModel  = model.fit(x = np.array(train_features), y = np.array(train_labels),class_weight=dict(enumerate(class_weights)), batch_size=20, epochs=500, shuffle=True, validation_split=.5) 

#get metrics

test_loss, test_accuracy = model.evaluate(np.array(test_features),  np.array(test_labels), verbose=2) 
print("test_loss: ", test_loss, "test_accuracy", test_accuracy)



from sklearn.metrics import classification_report
predictions = model.predict(np.array(test_features))
predicted_classes = np.argmax(predictions, axis=1)


classification_report = classification_report(test_labels, predicted_classes)
print(classification_report)

names = ["unacc", "acc", "good", "vgood"]
# for x in range(len(predicted_classes)):
    # print("Predicted: ", names[predicted_classes[x]],"actual", names[label[x]])



# mnistPred = model.predict(np.array(features))
# predicted_labels = np.argmax(mnistPred, axis=1)
# classification_report = classification_report(np.array(label), predicted_labels)
# print(classification_report)


