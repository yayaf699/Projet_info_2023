import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps, ImageDraw 
import smartcrop




# class_names = ['Zero', 'Un', 'Deux', 'Trois', 'Quatre', 'Cinq', 'Six', 'Sept', 'Huit', 'Neuf']

class_names = []

with open("classes.txt", 'r') as f:
    class_names = [line.rstrip('\n') for line in f]

print(class_names)

model = keras.models.load_model("draw2.h5")

# def normalize(link):
#     img = cv2.imread(link, 0)
#     #img = cv2.bitwise_not(img)
#     img = cv2.resize(img, (100, 100))
#     img = img / 255.0
#     img = np.expand_dims(img,0)
#     return img

# img = normalize("./test/test.png")

img = Image.open('/home/yanisse/Images/test.png')
img = ImageOps.grayscale(img)
img = img.resize( (28, 28), Image.Resampling.LANCZOS)
img = np.expand_dims(img,0)
plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.show()



# img = Image.open('/home/yanisse/Images/test.png')
# img = ImageOps.grayscale(img)
# # img = cv2.bitwise_not(img)
# img = img.resize( (28, 28), Image.Resampling.BILINEAR)
# # img = img / 255.0
# img = np.expand_dims(img,0)
# plt.figure()
# plt.imshow(img[0])
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(img[0])
# plt.colorbar()
# plt.show()

# def plot_value_array(i, predictions_array):
#   plt.grid(False)
#   plt.xticks(range(345))
#   plt.yticks([])
#   thisplot = plt.bar(range(345), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')

predictions_single = model.predict(img)

print(predictions_single)

print("J'ai predit :")
print(class_names[np.argmax(predictions_single)])


idx = (-predictions_single).argsort()[:3]
# top4 = predictions_single.argsort()

print(idx[0][0:3])

autre = 1 - predictions_single[0][idx[0][0]] - predictions_single[0][idx[0][1]] - predictions_single[0][idx[0][2]]
top3 = [predictions_single[0][idx[0][0]],
        predictions_single[0][idx[0][1]],
        predictions_single[0][idx[0][2]],
        autre]

print("top1: %s avec %.2f pourcents\ntop2: %s avec %.2f pourcents\ntop3: %s avec %.2f pourcents\nla probabilité pour les autres est %.2f pourcents"
         %(class_names[idx[0][0]],top3[0]*100,
        class_names[idx[0][1]],top3[1]*100,
        class_names[idx[0][2]],top3[2]*100,
        top3[3]*100))


# plot_value_array(1, predictions_single[0])
# _ = plt.xticks(range(345), class_names, rotation=45)
# plt.show()





plt.pie(top3, labels=(class_names[idx[0][0]],
                      class_names[idx[0][1]],
                      class_names[idx[0][2]],'others'),
                      autopct=lambda p:'{:.2f}%'.format(p))
plt.show()

