import cv2
import matplotlib.pyplot as plt

image = cv2.imread("test.jpg")
bins=(8, 8, 8)
hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
print(hist)

[plt.plot(histi) for histi in hist]
plt.show()