import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import cv2
import os

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def prepare_data(path):
    label = []
    image = []
    for x in os.listdir(path):
        if x.endswith('.jpg'):
            label.append(1 if x.split(".")[0] == "dog" else 0)

            f = os.path.join(path, x)
            image.append(extract_histogram(cv2.imread(f)))
    return image, label

def select_data_for_test(path, f_names):
    label = []
    image = []
    for x in f_names:
        label.append(1 if x.split(".")[0] == "dog" else 0)
        f = os.path.join(path, x)
        image.append([extract_histogram(cv2.imread(f))])
    return image, label

def main():
    rand = 13 # параметр random_state-а
    c = 1.28 # константа C
    teta = [0, 1, 32, 33, 34, 333, 334, 335] #индексы Тетты, которые спрашиваются
    test_images = ["cat.1043.jpg", "cat.1037.jpg", "dog.1012.jpg", "dog.1033.jpg"] #названия картинок из выборки для тестирования, для тестирования

    images, labels = prepare_data('data/train/')

    print(f"images = {np.shape(images)}, labels = {np.shape(labels)}")
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=int(rand))

    model = LinearSVC(random_state=int(rand), C=float(c))
    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    # print(model.coef_)
    [print(f"Teta[{ti}] = {round(model.coef_[0][ti], 3)}") for ti in teta]
    print(f"F1 = {round(f1_score(test_labels, predictions, average='macro'), 4)}")

    images, labels = select_data_for_test('data/test/', test_images)
    predictions = []
    for i in range(len(images)):
        prediction = model.predict(images[i])
        print(f"prediction {test_images[i]}: {prediction}")
        predictions.append(prediction[0])
    # print(f"Predicted value for img {}")

main()