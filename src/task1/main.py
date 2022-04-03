

import csv
import numpy as np

def print_data(data):
    for row in data:
        print(', '.join(row))

with open('data/ros_stat_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    data = list(spamreader)
    data = data[1:]
    selary = [int(row[1]) for row in data]

    #sort by the second row
    selary.sort()

    #enter for a, b, c your numbers from the first task + 1
    a1 = 21
    a2 = 22
    a3 = 77
    print(f'x_{a1} => {selary[a1 - 1]}')
    print(f'x_{a2} => {selary[a2 - 1]}')
    print(f'x_{a3} => {selary[a3 - 1]}')

    print(f'Выборочное среднее => {np.mean(selary)}')
    print(f'Выборочная медиана => {np.median(selary)}')
    csvfile.close()

# z = np.matrix([[3.17], [-1.58], [-1.59]])
# phi = np.matrix([0.32, 0.95])

# f = np.matmul(z, phi)
# print(f)
