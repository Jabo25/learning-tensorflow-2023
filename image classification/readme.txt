first iteration - (pt1)
I initially made an image classifier using a regular neural network without a convnet. This gave me around 10% accuracy with number classification.


second iteration - (pt2)_convnets
In my 2nd iteration of the model, I added a convnet to the neural network. This increased the accuracy of the model at the sacrifice of the speed 
each epoch could run. The accuracy of the model was 4-5% depending on the training run.

third iteration - (pt2)_data_augmentation
The 3rd iteration of the model was a modification of my first attempt, but I added data augmentation to the images and used random transformations
to create more training data. This increased the accuracy up to 2-3% after 30 epochs. The problem with this method is that the training time takes
much longer than the
