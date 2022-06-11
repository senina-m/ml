import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
import graphviz
import os


#пути на винде -- грустно
input_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "\\..\\data\\diabetes.csv", delimiter=',')
num_of_lines_to_cut =  560 # количество строк, которые нужно отделить в начале
data = input_data[:num_of_lines_to_cut]
print(f"Количество не больных диабетом в первых {num_of_lines_to_cut} строках:", data[data.Outcome == 0].shape[0])

y = data[['Outcome']]
x = data.drop(['Outcome'], axis=1)

print(f"x = {np.shape(x)}, y = {np.shape(y)}")
# print(x)
# print(y)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)

# criterion='entropy', max_leaf_nodes = 15, min_samples_leaf = 10 и random_state = 2020.
tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_leaf_nodes=15, random_state=2020)
clf = tree.fit(train_x, train_y)
print(f"Глубина дерева ", clf.tree_.max_depth)

# Graphviz (dot) language support for Visual Studio Code
columns = list(train_x.columns)
export_graphviz(clf, out_file='tree.dot', 
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

y_real = test_y['Outcome']
y_pred = clf.predict(test_x)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_real, y_pred)
print(f"Оценка точности модели = {round(accuracy_score, 5)}")

from sklearn.metrics import f1_score
f1_score = f1_score(y_real, y_pred, average='macro')
print(f"F1 = {round(f1_score, 5)}")
patients = [743, 707, 760, 704] #номера пациентов для проверки
for p in patients:
    patient = input_data.iloc[[p]]
    print(f"prediction for patient[{p}]=", clf.predict(patient.drop(['Outcome'], axis=1)))
