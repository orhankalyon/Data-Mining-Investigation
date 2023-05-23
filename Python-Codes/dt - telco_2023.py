import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filename = 'Datasets/telco_2023.csv'

data = pd.read_csv(filename)
classlabel = data.iloc[:, -1]
attr = data[['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
             'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill']]

#Spliting the data into training and testing sets.
attr_train, attr_test, class_train, class_test = train_test_split(attr, classlabel, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(attr_train, class_train)

predictions = model.predict(attr_test)

#Precision, Recall, F-score.
precision = metrics.precision_score(class_test, predictions)
recall = metrics.recall_score(class_test, predictions)
fscore = metrics.f1_score(class_test, predictions)
print('Precision:', precision)
print('Recall:', recall)
print('F-score:', fscore)