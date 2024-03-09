import cv2
import glob
import keras
import numpy
import numpy as np

model = keras.models.load_model("ricepredictor.keras")
model.summary()

dataset = []

afiles = r"Rice_Image_Dataset/Arborio/Arborio (1).jpg"
bfiles = r"Rice_Image_Dataset/Ipsala/Ipsala (4).jpg"

img1 = cv2.imread(afiles)
img1 = cv2.resize(img1,(250,250))
img2 = cv2.imread(bfiles)
img2 = cv2.resize(img2,(250,250))


img1 = np.expand_dims(img1,0)
img2 = np.expand_dims(img2,0)

print(model.predict(img1))
print(model.predict(img2))

