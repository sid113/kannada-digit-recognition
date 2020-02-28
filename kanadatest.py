import matplotlib.pyplot as plt
import cv2
img = cv2.imread('/home/zerome/Downloads/untitled.png',3)
plt.imshow(img)
plt.show()
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.show()

width = 28
height = 28
dim = (width, height)
 
# resize image
resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized)
plt.show()
print(resized.type)
