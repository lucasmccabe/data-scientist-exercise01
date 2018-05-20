from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from keras.utils import plot_model
import matplotlib.pyplot as pl
import seaborn as sns
import keras as K

model = K.models.load_model('model.h5')

ws = []
for layer in model.layers:
    weights = layer.get_weights()
    for i in weights:
        #print(i)
        for j in i:
            #print(j)
            ws.append(sum(abs(j)))
        #print(ws)
        break
    break

processed_data = pd.DataFrame.from_csv('processed_data.csv')
processed_data = processed_data.sample(frac=1).dropna() #this shuffles the row order and drops any rows with NaN values

out = []
for j in processed_data.std():
    out.append(j)

for i in range(len(ws)):
    print(ws[i]*out[i])

print(processed_data.std())