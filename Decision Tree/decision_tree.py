import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn import tree

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

#the decision tree tends to overfit the training data at the expense of lower accuracy in testing
#this can be mitigated by specifying a maximum tree depth

#using the validation data to pick a tree depth
validation_accs = [] #list of all validation accuracies
for i in range(1, 24):
    #decision tree model
    tree = DecisionTreeClassifier(max_depth=i,random_state=0) #specify model type
    tree.fit(data_train, targets_train) #fit the model

    #add accuracy for this test to testing_accs
    validation_accs.append(tree.score(data_validation, targets_validation))

depth = validation_accs.index(max(validation_accs))+1 #the tree depth that produced the greatest test accuracy
#in general this tends to be 7 or 8

print('\nSelecting a maximum tree depth of ' + str(depth))

#decision tree model
model = DecisionTreeClassifier(max_depth=depth,random_state=0) #specify model type
model.fit(data_train, targets_train) #fit the model

print('Testing accuracy: ' + str('%.2f'%(100*tree.score(data_test, targets_test))) + '%') #tends to be roughly 82%

#produce confusion matrix
confusion = pd.DataFrame(confusion_matrix(targets_test, tree.predict(data_test)),
    columns=['Predicted <=$50k', 'Predicted >$50k'],
    index=['True <=$50k', 'True >$50k'])

true_positive = confusion['Predicted >$50k']['True >$50k']
false_positive = confusion['Predicted >$50k']['True <=$50k']
false_negative = confusion['Predicted <=$50k']['True >$50k']
f1score = 2 * true_positive / (2 * true_positive + false_positive + false_negative) #calculate F1-score
print('F1-score: ' + str('%.4f'%f1score))
print('\nConfusion matrix: ')
print(confusion)

#save the decision tree
export_graphviz(model, out_file="decision_tree.dot", feature_names=list(processed_data.drop('over_50k', axis=1)), impurity=False, filled=True)

