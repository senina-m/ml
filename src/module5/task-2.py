import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

X = pd.read_csv("./data/candy-data.csv", delimiter=',')
n = len(X)

#удалили столбец Y вставили столбец с единицами
X = X.drop(columns=['Y'])
X.insert(loc=1, column='ones', value=([1]*n))

# выделили строки с тестовыми данными
test_candy_name1 = "One dime"
test_candy_name2 = "Nestle Butterfinger"
test_candy1 = X.loc[X['competitorname'] == test_candy_name1]
test_candy2 = X.loc[X['competitorname'] == test_candy_name2]
#стерли названия конфет и значения их рейтинга
candy1 = test_candy1.drop(columns=['competitorname', 'winpercent'])
candy2 = test_candy2.drop(columns=['competitorname', 'winpercent'])

#выкинули тестовые данные из общей выборки
x = X[~X['competitorname'].isin([test_candy_name1, test_candy_name2])]
n = n - 2

#стерли названия конфет
x = x.drop(columns=['competitorname'])
# np.array сразу без заголовков
# выделили столбец с y
y = np.array(x[['winpercent']]).reshape(1, n)
print(y)
print(np.shape(y))

#выделили строки под X
x = x.drop(columns=['winpercent'])


# x-> [68x12] y -> [1x68]
reg = LinearRegression().fit(x, y)
reg.fit(candy1, y)
reg.fit(candy2, y)



# print(f"Параметры регрессии b_0:", reg.intercept_)
# print(f"Параметры регрессии b_1:", reg.coef_)