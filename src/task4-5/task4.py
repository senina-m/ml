import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn

# file_name = имени вашего файла
train_data = pd.read_csv("./data/data.csv", delimiter = ",", index_col = "id")

X = pd.DataFrame(train_data.drop(['Class'], axis=1))
y = pd.DataFrame(train_data['Class']).values.ravel()

num_of_neighbors = 3
coordinate_x = 30
coordinate_y = 30

# Euclidian metric -> p = 2, n_neighbours = number of neighbors (k) 
euclid = knn(n_neighbors = num_of_neighbors, p = 2)
# manhattan_distance p = 1
manhattan = knn(n_neighbors = num_of_neighbors, p = 1)

euclid.fit(X, y)
manhattan.fit(X, y)

Object = [coordinate_x, coordinate_y]

print(f"Предсказанный класс для евклидовой метрики: ", euclid.predict([Object]))
print(f"Предсказанный класс для манхетенская метрики: ", manhattan.predict([Object]))

# ближайшие соседи для Евклидовой в порядке возрастания расстояний; 
# id элементов сдвинуто на 1, то есть, если во втором массиве вы получаете [1,2,3], то ответ будет: 2,3,4
print(f"Ближайшие соседи для евклидовой метрики: ", euclid.kneighbors([Object]))
print(f"Ближайшие соседи для манхетенской метрики: ", manhattan.kneighbors([Object]))
