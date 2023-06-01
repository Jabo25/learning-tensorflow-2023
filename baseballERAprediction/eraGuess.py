import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
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




# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
np.set_printoptions(precision=3, suppress=True)

#open the data
data = pd.read_csv("baseballERAprediction/editedStatsNo2020or2023.csv" )
print(data.head())
label = list(data.pop("p_era"))
a = data.pop("player_id")
a = data.pop("year")
a = data.pop("strikeout")
a = data.pop("walk")

a = data.pop("f_strike_percent")
a = data.pop("ff_avg_speed")
a = data.pop("fastball_avg_spin")
a = data.pop("breaking_avg_spin")
a = data.pop("offspeed_avg_spin")


k_percent = list(data.pop("k_percent"))

batting_avg = list(data.pop("batting_avg"))

exit_velocity_avg = list(data.pop("exit_velocity_avg"))

launch_angle_avg = list(data.pop("launch_angle_avg"))

barrel_batted_rate = list(data.pop("barrel_batted_rate"))

hard_hit_percent = list(data.pop("hard_hit_percent"))

avg_best_speed = list(data.pop("avg_best_speed"))

avg_hyper_speed = list(data.pop("avg_hyper_speed"))

whiff_percent = list(data.pop("whiff_percent"))

fastball_avg_speed = list(data.pop("fastball_avg_speed"))
print(data.head())







# k_percent,batting_avg,exit_velocity_avg,launch_angle_avg,barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hyper_speed,whiff_percent,f_strike_percent,ff_avg_speed,fastball_avg_speed,fastball_avg_spin,breaking_avg_spin,offspeed_avg_spin

# strikeout,walk,k_percent,batting_avg,p_era,exit_velocity_avg,launch_angle_avg,barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hyper_speed,whiff_percent,f_strike_percent,ff_avg_speed,fastball_avg_speed,fastball_avg_spin,breaking_avg_spin,offspeed_avg_spin



features =  list(zip(k_percent,batting_avg,exit_velocity_avg,launch_angle_avg,barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hyper_speed,whiff_percent,fastball_avg_speed))# ,data['k_percent'],data['batting_avg'],data['exit_velocity_avg'],data['launch_angle_avg'],data['barrel_batted_rate'],data['hard_hit_percent'],data['avg_best_speed,avg_hyper_speed'],data['whiff_percent'],data['f_strike_percent'],data['ff_avg_speed'],data['fastball_avg_speed'],data['fastball_avg_spin'],data['breaking_avg_spin'],data['offspeed_avg_spin']# features = list(zip(buying,maint,doors, persons,lug_boot,safety))# label = list(Class)


# #splits up the training and test set. 
train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, label, test_size = 0.1)



model = tf.keras.models.Sequential(layers.Dense(units=1))
# mpodel

test_results = {}
model.add(tf.keras.layers.Dense(10, activation='linear'))

model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['MSE' ])

# history = model.fit(train_features, train_labels,epochs = 100, batch_size = 20)
history = model.fit(np.array(train_features),np.array(train_labels),epochs=1000, verbose=1,validation_split = 0.2) 

MSE = model.evaluate(np.array(test_features), np.array(test_labels), verbose=2)
print(MSE)




predictions = model.predict(np.array(test_features))



#prints out all of the data. modifyable for column spicific data or noncolumn spicific data. 
difference = 0
total = 0
xCount = 0
for x in range(len(predictions)):
    # print("Predicted: ", predictions[x],"actual", label[x])
    difference = predictions[x]-test_labels[x]
    if difference<0:
        # print(difference)
        difference = difference*-1
    total = difference+total
    
    xCount = xCount+1
    xCount = float(xCount)
print(total, "xCount is", xCount)
average = 0.000
avgerage = float(difference)/float(xCount)


# print("the total average error is :", label)
    



# test_predictions = linear_model.predict(test_features)
# for prediction, score in zip(test_predictions, test_labels):
#     print(prediction, score)




# copy the dataset to a new array
# train_features = train_dataset.copy()
# test_features = test_dataset.copy()


# #declear which cols are going to be used for training
# train_labels = train_features.pop('G1')
# test_labels = test_features.pop('G1')


# #normalize the data andd add features to train on
# normalizer = np.array([train_features['school'],train_features['age'],train_features['Medu'],train_features['Fedu'],
#     train_features['traveltime'],train_features['studytime'],train_features['failures'],train_features['schoolsup'],
#     train_features['famsup'],train_features['paid'],train_features['activities'],train_features['nursery'],train_features['higher'],
#     train_features['internet'],train_features['romantic'],train_features['famrel'],train_features['freetime'],train_features['goout'],
#     train_features['Dalc'],train_features['Walc'],train_features['health'],train_features['absences']
# ])
# print(train_features.shape)
# # 
# normalizer = train_features.to_numpy()#makes numpy array
# good_normalizer = layers.Normalization( axis=-1)#makes input layer
# good_normalizer.adapt(normalizer)#makes


# #make a linear model and sequencial model 1 layer thick
# linear_model = tf.keras.Sequential([good_normalizer,layers.Dense(units=1)])
# print(linear_model.summary())

# #trains the model
# linear_model.predict(train_features[:10])#what we are training on
# linear_model.layers[1].kernel


# #model.compile configures the model and selects an optomizer
# linear_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')

# #model.fit trains the model
# history = linear_model.fit(train_features,train_labels,epochs=100, verbose=0,validation_split = 0.2)  # Suppress logging.  # Calculate validation results on 20% of the training data.

# test_results = {}


# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist)

# #plots the graph
# def plot_loss(history):
#   plt.plot(history.history['loss'], label='loss')
#   plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
#   plt.xlabel('Epoch')
#   plt.ylabel('Error [gpa]')
#   plt.legend()
#   plt.grid(True)
# plot_loss(history)


# #method applies the trained model to the test data and calculates the loss and accuracy.  Does stats but not do predictions
# test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=0)#verbose tells you how much output info will be shared
# print(test_results)


# # print(test_results['linear_model'])
# plt.show()


# #predicts the output using a model
# test_predictions = linear_model.predict(test_features)
# for prediction, score in zip(test_predictions, test_dataset['G1']):
#     print(prediction, score)