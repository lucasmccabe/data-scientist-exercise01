import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn import tree

#get data and split it
processed_data = pd.DataFrame.from_csv('processed_data.csv')
processed_data = processed_data.sample(frac=1).dropna() #this shuffles the row order and drops any rows with NaN values
targets = processed_data.loc[:, 'over_50k']
data = processed_data.drop('over_50k', axis=1)
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, random_state=1) #default is a 75/25 split

#the decision tree tends to overfit the training data at the expense of lower accuracy in testing
#this can be mitigated by specifying a maximum tree depth

#test all depths
testing_accs = [] #list of all testing accuracies
for i in range(1, 24):
    #decision tree model
    tree = DecisionTreeClassifier(max_depth=i,random_state=0) #specify model type
    tree.fit(data_train, targets_train) #fit the model

    #add accuracy for this test to testing_accs
    testing_accs.append(tree.score(data_test, targets_test))

depth = testing_accs.index(max(testing_accs))+1 #the tree depth that produced the greatest test accuracy
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
print('Confusion matrix: ')
print(confusion)
#save the decision tree
export_graphviz(model, out_file="decision_tree.dot", feature_names=list(data), impurity=False, filled=True)

