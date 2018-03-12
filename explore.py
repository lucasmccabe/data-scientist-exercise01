#initial data exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from time import time
%matplotlib inline

records = pd.DataFrame.from_csv('records.csv')
over_50 = records.loc[records['over_50k'] == 1] #all rows where income > $50k
under_50 = records.loc[records['over_50k'] == 0] #all rows where income <= $50k

#some general information about the dataset
print('Earning >$50k: ' + str(over_50.shape[0]) + ' individuals (' + str('%.2f'%(100*over_50.shape[0]/(over_50.shape[0]+under_50.shape[0]))) + '%)')
print('Earning <=$50k: ' + str(under_50.shape[0]) + ' individuals (' + str('%.2f'%(100*under_50.shape[0]/(over_50.shape[0]+under_50.shape[0]))) + '%)')
print('Summary of all records:')
print(records.describe())
print('Summary of >$50k records:')
print(over_50.describe())
print('Summary of <$50k records:')
print(under_50.describe())
print('-----')

#obervations re: capital gain/loss
print('Skewness of continuous variables:')
print(records.skew()) #capital gain and capital loss are highly skewed
