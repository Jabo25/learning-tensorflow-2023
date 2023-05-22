import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#opens the file
training_df = pd.read_csv(filepath_or_buffer="student-por.csv")
test_df = pd.read_csv(filepath_or_buffer="student-mat.csv")
# training_df["median_house_value"] /= 1000.0

training_df.head()




#build and train model (build_model(my_learning_rate), which builds a randomly-initialized model.
# training_df.build_model(.02)
# train_model(training_df, feature, label, epochs)
training_df.describe()



#@title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()                # Most simple tf.keras models are sequential.
    model.add(tf.keras.layers.Dense(units=1,            # Describe the topography of a simple linear regression model (single node in a single layer)
                                  input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",             # Compile the model topography into code that TensorFlow can efficiently execute. Configure training to minimize the model's mean squared error. 
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model        

def train_model(model, df, feature, label, my_epochs, 
                my_batch_size=None, my_validation_split=0.1):
  """Feed a dataset into the model in order to train it."""

  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=my_batch_size,
                      epochs=my_epochs,
                      validation_split=my_validation_split)

  # Gather the model's trained weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the 
  # rest of history.
  epochs = history.epoch
  
  # Isolate the root mean squared error for each epoch.
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse, history.history   

print("Defined the build_model and train_model functions.")
print("Defined the build_model and train_model functions.")
#@title Define the plotting functions

def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against 200 random training examples."""
  plt.xlabel(feature)# Label the axes.
  plt.ylabel(label)# Label the axes.
  random_examples = training_df.sample(n=200)# Create a scatter plot from 200 random points of the dataset.
  plt.scatter(random_examples[feature], random_examples[label])
  x0 = 0# Create a red line representing the model. The red line starts  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  y0 = trained_bias
  x1 = random_examples[feature].max()
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')
  plt.show()# Render the scatter plot and the red line.


def plot_the_loss_curve(epochs, mae_training, mae_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
  plt.legend()
  
  # We're not going to plot the first epoch, since the loss on the first epoch
  # is often substantially greater than the loss for other epochs.
  merged_mae_lists = mae_training[1:] + mae_validation[1:]
  highest_loss = max(merged_mae_lists)
  lowest_loss = min(merged_mae_lists)
  delta = highest_loss - lowest_loss
  print(delta)

  top_of_y_axis = highest_loss + (delta * 0.05)
  bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.show()  

print("Defined the plot_the_loss_curve function.")
print("Defined the plot_the_model and plot_the_loss_curve functions.")

learning_rate = 0.1
epochs = 50    
batch_size = 50
validation_split = 0.5




my_feature = "absences"
my_label = "G1"
my_model = None
print("after build_bodel reverse day :)")
my_model = build_model(learning_rate)
print("before build model")
shuffled_train_df = training_df.reindex(np.random.permutation(training_df.index))
epochs, rmse, history = train_model(my_model, training_df, my_feature, 
                                    my_label, epochs, batch_size, 
                                    validation_split)

# print("\nThe learned weight for your model is %.4f" % weight)
# print("The learned bias for your model is %.4f\n" % bias )/
# plot_the_model(weight, bias, my_feature, my_label)
# plot_the_loss_curve(epochs, rmse)
plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])
                    


def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][100:100 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][100 + i],
                                   training_df[label][100 + i],
                                   predicted_values[i][0] ))

predict_house_values(10,my_feature,my_label)

data = pd.DataFrame(training_df)
# print(data.corr())
x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)