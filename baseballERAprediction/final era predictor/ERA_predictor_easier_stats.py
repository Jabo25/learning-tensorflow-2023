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
data = pd.read_csv("baseballERAprediction/easyStats-allYear.csv" )
print(data.head())
label = list(data.pop("p_era"))

print(data.head())




k_percent = list(data['k_percent'])
batting_avg = list(data['batting_avg'])
whiff_percent  = list(data['whiff_percent'])
launch_angle_avg = list(data['launch_angle_avg'])
on_base_percent =  list(data['on_base_percent'])
on_base_plus_slg = list(data['on_base_plus_slg'])
slg_percent = list(data['slg_percent'])


#make an input list
features_unNormalized =  list(zip(k_percent,batting_avg,whiff_percent,launch_angle_avg, launch_angle_avg, on_base_percent,on_base_plus_slg, slg_percent))

def normalize_data(features_unNormalized):
# Compute mean and standard deviation
    mean = tf.reduce_mean(features_unNormalized, axis=0)
    std = tf.math.reduce_std(features_unNormalized, axis=0)

    # Normalize the data
    features = (features_unNormalized - mean) / std
    return features
features = normalize_data(features_unNormalized)


# #splits up the training and test set. 
train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(np.array(features), np.array(label), test_size=0.1)


model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear')])# model
test_results = {}
# model.add(tf.keras.layers.Dense(10, activation='linear'))


# model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['MSE' ])
model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['MSE' ])

# history = model.fit(train_features, train_labels,epochs = 100, batch_size = 20)
history = model.fit(np.array(train_features),np.array(train_labels),batch_size=50 , epochs=100 , verbose=1,validation_split = 0.2) 
# history = model.fit(np.array(train_features),np.array(train_labels),epochs=1000, verbose=1,validation_split = 0.2) 


MSE = model.evaluate(np.array(test_features), np.array(test_labels), verbose=2)
print(MSE)



#prepare the data for predictions
input_data = pd.read_csv("baseballERAprediction/easyStats_partial_season.csv" )
input_label = list(input_data.pop("p_era"))
print(input_data.head())

#get the input dealt with. 
input_k_percent = list(input_data['k_percent'])
input_batting_avg = list(input_data['batting_avg'])
input_whiff_percent  = list(input_data['whiff_percent'])
input_launch_angle_avg = list(input_data['launch_angle_avg'])
input_on_base_percent =  list(input_data['on_base_percent'])
input_on_base_plus_slg = list(input_data['on_base_plus_slg'])
input_slg_percent = list(input_data['slg_percent'])

#output tracker items; 
first_name = list(input_data.pop(" first_name"))
last_name = list(input_data.pop("last_name"))
year = list(input_data.pop("year"))





#do the predictions
predict_unNormalized =  list(zip(input_k_percent,input_batting_avg,input_whiff_percent,input_launch_angle_avg, input_launch_angle_avg, input_on_base_percent,input_on_base_plus_slg, input_slg_percent))
prediction_features = normalize_data(predict_unNormalized)
predictions = model.predict(np.array(prediction_features))

error = []
for x in range(len(predictions)):
    print( "predicted:", predictions[x] , "actual: ", input_label[x], year[x], first_name[x],last_name[x])
    error.append(abs(input_label[x]-predictions[x]))
    if(predictions[x]-input_label[x]<-1):
        print("anomonly-----------------")

# Ensure predictions and error are 1D arrays
predictions = np.squeeze(predictions)
error = np.squeeze(error)

# Create scatter plot
plt.scatter(predictions, error, marker='o')

# Fit a polynomial regression model
coefficients = np.polyfit(predictions, error, 1)
polynomial = np.poly1d(coefficients)
x = np.linspace(min(predictions), max(predictions), 100)
y = polynomial(x)

# Plot the line of best fit
plt.plot(x, y, color='red')

# Set labels and title
plt.xlabel('Predicted ERA')
plt.ylabel('Total Error')
plt.legend(['Predicted vs Actual ERA'])
plt.grid(True)

# Display the plot
plt.show()
# print(year, first_name, last_name)


#  Create scatter plot
plt.scatter(input_launch_angle_avg, predictions, marker='o')

# Fit a polynomial regression model
coefficients = np.polyfit(input_launch_angle_avg, predictions, 1)
polynomial = np.poly1d(coefficients)
x = np.linspace(min(input_launch_angle_avg), max(input_launch_angle_avg), 100)
y = polynomial(x)

# Plot the line of best fit
plt.plot(x, y, color='red')

# Set labels and title
plt.xlabel('launch angle')
plt.ylabel('Predicted ERA')
plt.legend(['launch angle vs Predicted ERA'])
plt.grid(True)

# Display the plot
plt.show()




#  Create scatter plot
plt.scatter(input_on_base_plus_slg, predictions, marker='o')

# Fit a polynomial regression model
coefficients = np.polyfit(input_on_base_plus_slg, predictions, 1)
polynomial = np.poly1d(coefficients)
x = np.linspace(min(input_on_base_plus_slg), max(input_on_base_plus_slg), 100)
y = polynomial(x)

# Plot the line of best fit
plt.plot(x, y, color='red')

# Set labels and title
plt.xlabel('on base plus slugging')
plt.ylabel('Predicted ERA')
plt.legend(['OPS vs Predicted ERA'])
plt.grid(True)

# Display the plot
plt.show()
