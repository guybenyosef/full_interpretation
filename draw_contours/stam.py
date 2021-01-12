import cv2
import numpy as np
from matplotlib import pyplot as plt

## Read and convert
#img = io.imread('http://matlabtricks.com/images/post-35/man.png')
img = cv2.imread('man.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## Find outer contours
_, cnts, _= cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

## Draw
canvas = np.zeros_like(img)
cv2.drawContours(canvas , cnts, -1, (0, 255, 0), 1)

plt.imshow(canvas)
plt.show()
#input()
