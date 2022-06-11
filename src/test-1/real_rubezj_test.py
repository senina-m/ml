import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn

df = pd.read_csv("pulsar_stars_new.csv", delimiter=',')

v = df[((df['TG'] == 1) & (df['MIP'] >= 52.9296875) & (df['MIP'] <= 58.7890625)) | ((df['TG'] == 0) & (df['MIP'] >= 87.5859375) & (df['MIP'] <= 88.484375))]
# print(v)

print(f"len: ", len(v))
mip = v[['MIP']]
print(f"mean MIP: ", np.mean(np.asarray(mip)))

x = v.drop(['TG'], axis=1)
y = v[['TG']]

# scaler = StandardScaler(with_std=False).fit(x)
# # print(scaler.mean_)
# s_x = scaler.transform(x)
# print(s_x)

s_x = (x - x.min())/(x.max() - x.min())

print(f"mean MIP after scaling: ", np.mean(np.asarray(s_x['MIP'])))
# print(s_x)


reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(s_x, y.values.ravel())

star = [[0.833, 0.092, 0.443, 0.092, 0.112, 0.86, 0.742, 0.299]]

print(f"prediction:", reg.predict(star))
print(f"prediction probability:", reg.predict_proba(star))

euclid = knn(p = 2).fit(s_x, y.values.ravel())

print(f"neighbours: ", euclid.kneighbors(star))


