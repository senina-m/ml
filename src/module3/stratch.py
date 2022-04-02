import numpy as np
from numpy import linalg as LA
import csv
import matplotlib.pyplot as plt

with open('data/34_16.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    data = list(spamreader)
    data = [[float(i) for i in row] for row in data]
    data = np.array(data)
    n = len(data)
    p = len(data[0])

    
    #нормируем данные
    mean = np.mean(data, axis=0)
    # print(f"mean values in each column: ", mean)
    normed_data = np.array([(row - mean) for row in data])

    #дальше надо посчитать ковариационную матрицу "тета" как 1/n * (F * transpose(F))
    teta = 1/n * np.matmul(normed_data, normed_data.transpose())

    #но проще посчитать это вот так:
    # teta = np.cov(normed_data)
    
    #находим собственные числа и векторы теты
    #числа        #векторы
    eigenvalues, eigenvectors = LA.eig(teta)

    # +++++++++++++++++++
    #дальше надо найти сколько мы будем оставлять главных компонент
    #построим график зависимости доли необъяснённой дисперсии от количества взятых компонент
    plt.plot(range(0, p), [1 - sum(eigenvalues[:i]/sum(eigenvalues)) for i in range(0, p)])
    plt.ylabel("доля необъяснённой дисперсии")
    plt.xlabel("k")
    plt.show()

    #дальше остаётся только найти значения наших данных в новых координатах
    #т.е вычислить Z = F*Phi, где составлен из собственных векторов тета, причём
    # в первом столбце Phiстоят координаты вектора весов, отвечающего наибольшему
    # собственному числу матрицы Phi, во втором – координаты вектора весов,
    # и так далее по убыванию

    # values + vectors
    eigen_matrix = np.array([np.array([eigenvalues[i], eigenvectors[i]]) for i in range(0, len(eigenvalues))])
    eigen_matrix = sorted(eigen_matrix, key=lambda x: x[0], reverse=True)

    #надо расставить вектора из eigen_matrix как столбцы матрицы по порядку, по которому они идут вeigen_matrix
    z  = []
    # ++++++++++++++++++++++++

    # вот мы  получим значения, но где-то между +++ и +++ ошибка


    csvfile.close()