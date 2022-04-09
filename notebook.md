# Пакеты
Пакет для логичтическая реграссии ```from sklearn.linear_model import LogisticRegression```

Пакет с разными метриками ```from sklearn import metrics```

Пакет для метода ближайших соседей ```from sklearn.neighbors import KNeighborsClassifier as knn```


# Pandas

**Прочитать данные из файла с помощью pandas** (Разделитель ```delimiter```, колонка которая будет использованная как имена строк ```index_col```,
если  у данных нет заголовка пишем  ```header=None```)

```DATA = pd.read_csv("filename.csv", delimiter=',', index_col='competitorname', header=None)```

Удалить строку с названием 
```data.drop([ "row_name1", "row_name2", "row_name3"])```

 Удалить столбцы с названиями
``` data.drop(['column1', 'column2'], axis=1)```

Сделать новый dataframe
```X = pd.DataFrame(data for dataframe)```

Выбрать только столбцы с названиями
```data[['column1', 'column2']]```

Выбрать строки с названиями
```data['row1', 'row2']```

Выделить строку со значением в столбце ```colum_1``` равным данному ```name_1```:
```row = X.loc[X['colum_1'] == 'name_1']```

Выбрать из X все строки, у которых в столбце ```competitorname``` стоит не ```name1``` или ```name1```:
```x = X[~X['competitorname'].isin(['name1', 'name2'])]```

Данные без подписей столбцов и строк
```data_frame.values```

Данные как массив без подписей 
```data_frame.values.ravel()```

# Numpy
reshape
```np.asarray(data).reshape(n, 1)```

Среднее ```x_mean = np.mean(x)```

#LinearRegression

Размерность входных данных для линейной регрессии 

```x-> [num_of_rows x num_of_columns] y -> [num_of_rows x 1]```

```reg = LinearRegression().fit(x, y)```

R квадрат статистка ```r2 = reg.score(x, y)```

Параметры регрессии b_0: ```reg.intercept_```

Параметры регрессии b_1: ```reg.coef_```

#LogisticRegression
```reg = LogisticRegression(random_state=, solver='lbfgs').fit(X, y)```

Предсказание с помощью обученной модели, порог отсечения по умолчанию составляет 0.5 ```Y_pred = reg.predict(X_test)```

Предсказать вероятности оценок ```Y_pred_probs = reg.predict_proba(X_test)```

```predict_proba()``` возвращает таблицу, где каждая строка соответствует ```i``` объекту выборки, а каждый столбец к ```j``` классу

Пример: `For candy {question_candy_name1} probability of 0 = {Y_pred_probs[candy1_number][0]}`

```
Y_pred_probs_class_1 = Y_pred_probs[:, 1]
print(Y_pred_probs_class_1)

Y_true = (test_data['Y'].to_frame().T).values.ravel()
fpr, tpr, _ = metrics.roc_curve(Y_true, Y_pred)
вычисляем AUC
print(f"AUC: ", metrics.roc_auc_score(Y_true, Y_pred_probs_class_1))
вычисление Recall
print(f"Recall: ", metrics.recall_score(Y_true, Y_pred))
вычисление Precision
print(f"Precision: ", metrics.precision_score(Y_true, Y_pred))
```

Посчитать ```accuracy```: ```reg.score(X, y)```

#KNeighborsClassifier

Обучение модели методом наименьших соседей с Манхеттанской  ```(p = 1)``` и Евклидовой метриками```(p = 2)```

```euclid = knn(n_neighbors = num_of_neighbors, p = 2)``` `euclid.fit(X, y)`

```manhattan = knn(n_neighbors = num_of_neighbors, p = 1)``` ```manhattan.fit(X, y)```

```Object = [coordinate_x, coordinate_y]```
Предсказанный класс для евклидовой метрики: ```euclid.predict([Object])```
Предсказанный класс для манхетенская метрики: ```manhattan.predict([Object])```

Ближайшие соседи для Евклидовой в порядке возрастания расстояний

id элементов сдвинуто на 1, то есть, если во втором массиве вы получаете [1,2,3], то ответ будет: 2,3,4

Ближайшие соседи для евклидовой метрики: ```euclid.kneighbors([Object])```