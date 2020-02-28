import matplotlib.pyplot as plt
from keras.models import load_model

import cv2
model = load_model('/home/zerome/weights00000025.h5')


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)
   


img = cv2.imread('/home/zerome/Downloads/sunflower/31.jpg',3)
plt.imshow(img)
plt.show()


width = 224
height = 224
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized)
plt.show()
img = resized.reshape(1, 224, 224, 3)
digit = model.predict_classes(img)
print(digit)
