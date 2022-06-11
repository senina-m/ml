import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn

data = pd.read_csv('./data/pulsar_stars_new.csv', delimiter=',', decimal='.')

v = data[(data.MIP >= 10) & (data.MIP <= 100)]
print(f"len: ", len(v))
print(f"mean in MIP: ", round(v.MIP.mean(), 3))
print(f"SIP max: ", data.SIP.max())

y = v[["TG"]]
x = v.drop(['TG'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
print(f"max in SIP after split", x_train.SIP.max())

scaler = StandardScaler()
s_v_train = scaler.fit(x_train, y_train)
s_x_train = scaler.transform(x_train)
s_x_test = scaler.transform(x_test)
print(f"max after scale", np.max(s_x_train, axis=0))


reg = LogisticRegression(random_state=2022, solver='lbfgs').fit(s_x_train, y_train)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"coefficients:", reg.coef_)
obj = [[136.750000, 57.178449, -0.068415, -0.636238, 3.642977, 20.959280, 6.896499, 53.593661]]
s_obj = scaler.transform(obj)

print(f"Accuracy: ", reg.score(s_x_test, y_test))
print(reg.predict_proba(scaler.transform(obj))[0][reg.predict(s_obj)[0]])

euclid = knn(n_neighbors=5, p=2).fit(s_x_train, y_train)

print(f"knn prediction:", euclid.predict(s_obj))
print(euclid.kneighbors(s_obj))

