import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from googletrans import Translator


class_names = []

with open("classes.txt", 'r') as f:
    class_names = [line.rstrip('\n') for line in f]

print(class_names)

def transform_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    dst = cv2.Canny(gray, 0, 150)
    blured = cv2.blur(dst, (5,5), 0)    
    MIN_CONTOUR_AREA=200
    img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    Contours,imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in Contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [X, Y, W, H] = cv2.boundingRect(contour)
    for i in range(20):
        if Y > 0 and X > 0:
            Y=Y-1
            X=X-1
        if Y+H <img.shape[1] and X+H < img.shape[0]:
            H=H+2
            W=W+2
    img = img[Y:Y+H, X:X+W]
    img = Image.fromarray(img)
    img = ImageOps.grayscale(img)
    img = img.resize( (28, 28), Image.Resampling.LANCZOS)
    return np.expand_dims(img,0)

model = keras.models.load_model("model_draw.h5")

img = Image.open("lien_vers_l_image")

img = transform_img(img)

predictions_single = model.predict(img)

print("J'ai predit :")
print(class_names[np.argmax(predictions_single)])


idx = (-predictions_single).argsort()[:3]

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

