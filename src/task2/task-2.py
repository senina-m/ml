import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_redused = np.matrix(pd.read_csv("data/X_reduced_792.csv", delimiter=';'))
X_loadings = np.matrix(pd.read_csv("data/X_loadings_792.csv", delimiter=';'))

f = np.dot(X_redused, np.transpose(X_loadings))
plt.imshow(f)
plt.show()