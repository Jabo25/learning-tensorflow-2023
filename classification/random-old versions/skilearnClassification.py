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
from sklearn.metrics import classification_report

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.metrics import classification_report
np.set_printoptions(precision=3, suppress=True)



# #create dataframe
data = pd.read_csv('classification/car.csv')
print(data.head())


# #transformas all fo the buying values into a list of apropriate int encoded values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
door = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
buying = le.fit_transform(list(data["buying"]))






predict = "class"

#isolates the features and training data
X = list(zip(buying,maint,door, persons,lug_boot,safety))
y = list(cls)
# X = list(zip(buying, maint, door, persons, lug_boot, safety))
# y = list(cls)
# #split the training and the test data up
# feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(features, label, test_size = 0.1)
# model = KNeighborsClassifier(n_neighbors=9)

# #trains the model and scores outpuut
# model.fit(feature_train, label_train)
# acc = model.score(feature_test, label_test)
# print(acc)


# #makes predictions
# predicted = model.predict(feature_test)

# #decoder
# names = ["unacc", "acc", "good", "vgood"]
# for x in range(len(predicted)):
#     print("Predicted: ", names[predicted[x]],"actual", names[label_test[x]])
#     # n = model.kneighbors([feature_test[x]], 9, True)
#     # print("N: ", n)



# # for x in range(len(predicted)):
# #     print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
# #     n = model.kneighbors([x_test[x]], 9, True)
# #     print("N: ", n)
# # classification_report = classification_report(y_test, predicted)
# # print(classification_report)




# classification_report = classification_report(label_test, predicted)
# print(classification_report)


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.ops.gen_data_flow_ops import fifo_queue
# from tensorflow.keras import layers
# from matplotlib import pyplot as plt
# from keras import backend as K
# from sklearn.metrics import classification_report

# import sklearn
# from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import linear_model, preprocessing
# import sklearn
# from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import linear_model, preprocessing
# from sklearn.metrics import classification_report
# np.set_printoptions(precision=3, suppress=True)



#create dataframe
# data = pd.read_csv('classification/car.csv')
# print(data.head())
# le = preprocessing.LabelEncoder()
# buying = le.fit_transform(list(data["buying"]))
# maint = le.fit_transform(list(data["maint"]))
# door = le.fit_transform(list(data["doors"]))
# persons = le.fit_transform(list(data["persons"]))
# lug_boot = le.fit_transform(list(data["lug_boot"]))
# safety = le.fit_transform(list(data["safety"]))
# cls = le.fit_transform(list(data["class"]))

# predict = "class"

# X = list(zip(buying, maint, door, persons, lug_boot, safety))
# y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
classification_report = classification_report(y_test, predicted)
print(classification_report)


