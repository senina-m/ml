# **Основные методы и пакеты курса**

# Пакеты
Пакет для логичтическая реграссии ```from sklearn.linear_model import LogisticRegression```

Пакет с разными метриками ```from sklearn import metrics```

Пакет для метода ближайших соседей ```from sklearn.neighbors import KNeighborsClassifier as knn```

Пакет для дерева решений ```from sklearn.tree import export_graphviz```

Пакет для разделения выборки на тестовую и обучающую ```from sklearn.model_selection import train_test_split```

Метрика F1 из уражнения 7 ```from sklearn.metrics import f1_score```

Пакет для метода опорных векторов ```from sklearn.svm import LinearSVC```



# Pandas

**Прочитать данные из файла с помощью pandas** (Разделитель ```delimiter```, колонка которая будет использованная как имена строк ```index_col```,
если  у данных нет заголовка пишем  ```header=None```)

```DATA = pd.read_csv("filename.csv", delimiter=',', index_col='competitorname', header=None)```

```data = pd.DataFrame(q_list, columns=['q_data'])``` (q_list = [1, 2, 3, ..., 10])
```data = pd.DataFrame(array[1:], columns=array[0])``` (q_list = [[name_1, name_2, ...], [], ... []])

## Подсчёт значений и условия

Количество строк в df
```data.shape[0]```, количество строк с условием ```df[df.A > 0].shape[0]```

Строки подходящие под условие значение в столбце A > 0
```df[df.A > 0].shape[0]```

Если надо указать несколько условий
```df[(df['A'] > 0) & (df['B'] > 0)]```

## Изменение df

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

Строку под номером i не считая заголовков:
```df.iloc[[i]]```

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

# LogisticRegression
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

# KNeighborsClassifier

Обучение модели методом наименьших соседей с Манхеттанской  ```(p = 1)``` и Евклидовой метриками```(p = 2)```

```euclid = knn(n_neighbors = num_of_neighbors, p = 2)``` `euclid.fit(X, y)`

```manhattan = knn(n_neighbors = num_of_neighbors, p = 1)``` ```manhattan.fit(X, y)```

```Object = [coordinate_x, coordinate_y]```
Предсказанный класс для евклидовой метрики: ```euclid.predict([Object])```

Предсказанный класс для манхетенская метрики: ```manhattan.predict([Object])```

Ближайшие соседи для Евклидовой в порядке возрастания расстояний

id элементов сдвинуто на 1, то есть, если во втором массиве вы получаете [1,2,3], то ответ будет: 2,3,4

Ближайшие соседи для евклидовой метрики: ```euclid.kneighbors([Object])```


# LinearSVC опорные вектора

Извлечение гистограммы из картинки
```
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
```
Построение модели по методу опорных векторов
```model = LinearSVC(random_state=int(rand), C=float(c))```

Обучение модели
```model.fit(train_data, train_labels)```

Применение обученой модели на полученных данных
```predictions = model.predict(test_data)```

Коэффициенты модели ```model.coef_```

Применение метрики F1
```f1_score(test_labels, predictions, average='macro')```

Применение метрики точности предсказания  
```from sklearn.metrics import accuracy_score```

# os отфильтровать картинки по названию
Директория из которой запустили скрипт ```os.path.dirname(os.path.abspath(__file__))```

Все имена файлов директории с данным путём ```os.listdir(path)```

Заканчивается ли строка х на .jpg ```x.endswith('.jpg')```

Файл с путём до директории ```path``` и именем файла ```x```: ```f = os.path.join(path, x)```

Извлечение гистограммы из картинки с помощью пакета ```cv2```: ```extract_histogram(cv2.imread(f))```

# train_test_split()
Пример разделения выборки на обучающую и тестовую 75% к 25% рандомным способом с рандомизацией ```rand```

Тут m предикторов это массив ```images[n x m]``` и отклик это ```labes[1 x n]```  :

```train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=int(rand))```

Если не нужно перемешивать данные:

```train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)```

# DecisionTreeClassifier



## Визуализация дерева .dot

Чтобы визуализиовать дерево из метода дерева решений можно воспользоваться этим кодом из упражнения:

```
from sklearn.tree import export_graphviz
import graphviz
columns = list(train_x.columns)
export_graphviz(clf, out_file='tree.dot', 
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)
```

В итоге, запуск программы сгенерит файл tree.dot который можно открыть (ctrl + shift + v в vscode) 

и посмотреть на само дерево глазами, если устновить дополнение: *Graphviz (dot) language support for Visual Studio Code*



