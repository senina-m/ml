import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os


#пути на винде -- грустно
data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "\\..\\data\\data.csv", delimiter=',', index_col='Object')

prediction = data[['Cluster']]
data_to_cluster = data.drop(['Cluster'], axis=1)

centroid = np.array([[7.5, 12.0], [11.43, 8.43], [11.25, 9.75]])
kmeans = KMeans(n_clusters=3, init=centroid, max_iter=100, n_init=1)
model = kmeans.fit(data_to_cluster)

answers = model.labels_.tolist()
print(answers)

[print(f"item {i}  => {answers[i]}") for i in range(len(answers))]

dist = kmeans.fit_transform(data_to_cluster)

my_claster = []

for i in range(len(dist)):
    if answers[i] == 0:
        my_claster.append(dist[i][0].tolist())

print(f"mean dist to centroid = ", round(np.mean(my_claster), 3))