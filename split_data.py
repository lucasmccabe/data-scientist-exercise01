import numpy as np
import pandas as pd

#get data and split it
data = pd.DataFrame.from_csv('processed_data.csv')
data = data.sample(frac=1) #this shuffles the row order
train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))]) #using a 60/20/20 split
print('Training data shape: ' + str(train.shape))
print('Validation data shape: ' + str(validate.shape))
print('Testing data shape: ' + str(test.shape))