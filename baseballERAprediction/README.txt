I wanted to see how well I could predict pitcher ERA in the MLB by using a basic machine-learning model. 
All of my data was collected from baseball savant which is ran by the MLB. https://baseballsavant.mlb.com/
Baseball Savant allowed me to download a CSV file of every pitcher statistic in the Statcast era of baseball. This data is powered by a model. 
I have a link to my google colab to make it easier to run: https://colab.research.google.com/drive/1PJAej-CpBxssjjNKtxBxK9N5Ua2oYa2H?usp=sharing#scrollTo=HIQdkZ80QedD

There are 2 versions of the project I used. The statcast-only model uses Statcast era statistics like launch angle and hard hit rate to predict ERA. 
However, it does not have older statistics like OPS, SLG, and OBP. I also did not show graphs or use normalization in this model. 
The statcast-only model has an accuracy of .4 away from pitcher ERA on average which is good, but it could be improved. 

The 2nd version of the model I made, and the better version of the model is in the folder titled FINAL ERA PREDICTOR. This version of the algorithm is
.28 away on average from the actual pitcher ERA. This model uses Statcast statistics like launch angle as well as older statistics like OPS and SLG%.
I also added normalization through a zscore which significantly improved model accuracy, speed, and performance when running it. 


Note that the data I tested the algorithm on was from a shortened season, and it is entirely possible that the model would be more accurate with a
full season of stats. However, there is no benefit in predicting pitcher performance after an MLB season has ended. The algorithm favors strikes and
pitchers especially OPS as shown in my graphs. It predicts Sandy Alcantara is underperforming by a longshot and should have an ERA of in the mid 3's,
and that Tanner Houck is also underperforming and should have an ERA in the mid 3'srather than the mid 5's. 

Interestingly, Bryce Elder's ERA is very low, and it is not explained by any conventional statistics I can find. I think that he is a very good pitcher,
but his performance can only be explained by more advanced stat cast statistics my model is not accounting for. His style of pitching seems to be different 
and unconventional comparedto other elite MLB pitchers. 
