import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

DATA = pd.read_csv("data/candy-data.csv", delimiter=',', index_col='competitorname')

# Candies that are unused in training
train_data = DATA.drop([ "Nestle Crunch", "Skittles wildberry", "Sour Patch Tricksters"])

X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
y = pd.DataFrame(train_data['Y'])

# параметры random_state и solver даны в задании
reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, y.values.ravel())
test_data = pd.read_csv("data/candy-test.csv", delimiter=',', index_col='competitorname')
X_test = pd.DataFrame(test_data.drop(['Y'], axis=1))

# предсказание с помощью обученной модели, порог отсечения по умолчанию составляет 0.5
Y_pred = reg.predict(X_test)
Y_pred_probs = reg.predict_proba(X_test)

question_candy_name1 = "Werthers Original Caramel"
question_candy_name2 = "Sugar Babies"

test = pd.read_csv("data/candy-test.csv", delimiter=',')
candy1_number = test[test['competitorname'] == question_candy_name1].index.values.astype(int)[0]
candy2_number = test[test['competitorname'] == question_candy_name2].index.values.astype(int)[0]

print(f"For candy {question_candy_name1} probability of 0 = {Y_pred_probs[candy1_number][0]}")
print(f"For candy {question_candy_name1} probability of 1 = {Y_pred_probs[candy1_number][1]}")
print(f"For candy {question_candy_name2} probability of 0 = {Y_pred_probs[candy2_number][0]}")
print(f"For candy {question_candy_name2} probability of 1 = {Y_pred_probs[candy2_number][1]}")


Y_pred_probs_class_1 = Y_pred_probs[:, 1]
# print(Y_pred_probs_class_1)

Y_true = (test_data['Y'].to_frame().T).values.ravel()
fpr, tpr, _ = metrics.roc_curve(Y_true, Y_pred)
# вычисляем AUC
print(f"AUC: ", metrics.roc_auc_score(Y_true, Y_pred_probs_class_1))
# вычисление Recall
print(f"Recall: ", metrics.recall_score(Y_true, Y_pred))
# вычисление Precision
print(f"Precision: ", metrics.precision_score(Y_true, Y_pred))
