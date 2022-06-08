import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

array = [["A", "B", "C", "D"],
        [1, 2, 3, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 0],
        [10, 11, 12, 1],
        [15, 14, 13, 1],]

data = pd.DataFrame(array[1:], columns=array[0])
y = data[['D']]
x = data.drop(['D'], axis=1)

print(f"x = {np.shape(x)}, y = {np.shape(y)}")
print(x)
print(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)
print(f"tr_x\n", train_x)
print(f"tr_y\n", train_y)
print(f"ts_x\n", test_x)
print(f"ts_y\n", test_y)