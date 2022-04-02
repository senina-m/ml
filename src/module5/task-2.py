import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

X = pd.read_csv("./data/candy-data.csv", delimiter=',')
n = len(X)

#удалили столбец Y вставили столбец с единицами
X = X.drop(columns=['Y'])
# X.insert(loc=1, column='ones', value=([1]*n))
p = X.shape[1] - 1

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
p -= 1

# np.array сразу без заголовков
# выделили столбец с y
y = np.asarray(x[['winpercent']]).reshape(n, 1)

#выделили строки под X
x = x.drop(columns=['winpercent'])


reg = LinearRegression()
# x-> [68x12] y -> [68x1]
reg.fit(x, y)
print(f"{test_candy_name1} prediction:", reg.predict(candy1))
print(f"{test_candy_name2} prediction:", reg.predict(candy2))
test_vector = np.asarray([0, 1, 1, 1, 1, 1, 1, 0, 1, 0.26, 0.296]).reshape(1, p)
# print(test_vector)
print(f"test vector prediction:", reg.predict(test_vector))



# print(f"Параметры регрессии b_0:", reg.intercept_)
# print(f"Параметры регрессии b_1:", reg.coef_)