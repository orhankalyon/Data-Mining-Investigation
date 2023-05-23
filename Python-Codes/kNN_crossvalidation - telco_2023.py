import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn import metrics
import matplotlib.pyplot as plt

filename = 'Datasets/telco_2023.csv'

data = pd.read_csv(filename)
classlabel = data.iloc[:, -1]
attr = data[['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
             'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill']]

k=10
kf = KFold(n_splits=k, random_state=None)
model = KNeighborsClassifier(n_neighbors=5)

acc_score = []

for train_index , test_index in kf.split(attr):
    X_train , X_test = attr.iloc[train_index,:], attr.iloc[test_index,:]
    y_train , y_test = classlabel[train_index] , classlabel[test_index]
     
    model.fit(X_train, y_train)
    pred_values = model.predict(X_test)
     
    acc = metrics.accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('Accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))