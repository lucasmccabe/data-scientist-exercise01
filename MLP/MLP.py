from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd


#get data and split it into training, validation, and testing sets
processed_data = pd.DataFrame.from_csv('processed_data.csv')
processed_data = processed_data.sample(frac=1).dropna() #this shuffles the row order and drops any rows with NaN values
train, validation, test = np.split(processed_data, [int(.6*len(processed_data)), int(.8*len(processed_data))]) #uses a 60/20/20 split
data_train = train.drop('over_50k', axis=1)
data_validation = validation.drop('over_50k', axis=1)
data_test = test.drop('over_50k', axis=1)
targets_train = train.loc[:, 'over_50k']
targets_validation = validation.loc[:, 'over_50k']
targets_test = test.loc[:, 'over_50k']

#custom F1-score metric
def f1score(targets_true, targets_pred):
    # produce confusion matrix
    confusion = pd.DataFrame(confusion_matrix(targets_true, targets_pred),
                             columns=['Predicted <=$50k', 'Predicted >$50k'],
                             index=['True <=$50k', 'True >$50k'])

    #F1-score
    true_positive = confusion['Predicted >$50k']['True >$50k']
    false_positive = confusion['Predicted >$50k']['True <=$50k']
    false_negative = confusion['Predicted <=$50k']['True >$50k']
    f = 2 * true_positive / (2 * true_positive + false_positive + false_negative)  #formula for F1-score

    return f

#model architecture is simple, with 24 input neurons, 8 hidden neurons, and one output neuron
model = Sequential()
model.add(Dense(8, input_dim=24, activation='relu')) #hidden layer
model.add(Dense(1, activation='sigmoid')) #output layer

#compile model
model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

#fit model
model.fit(data_train, targets_train, validation_data=(data_validation, targets_validation), nb_epoch=10, batch_size=64)

#output testing accuracy and F1-score
#get predictions on test data
predictions = np.rint(model.predict(data_test)) #the predictions are continuous, so I round them to the nearest integer (0 or 1)
print('\nTesting accuracy: ' + str('%.2f'%(100*accuracy_score(targets_test, predictions))) + '%') #typically around 85%
print('F1-score: ' + str('%.4f'%f1score(targets_test, predictions))) #typicaly around 0.64

#save model weights
model.save_weights("model.h5")