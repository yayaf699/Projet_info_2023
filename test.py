import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2

class_names = ['Zero', 'Un', 'Deux', 'Trois', 'Quatre', 'Cinq', 'Six', 'Sept', 'Huit', 'Neuf']


model = keras.models.load_model("model3.h5")

def normalize(link):
    img = cv2.imread(link, 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img,0)
    return img


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

img = normalize("./test/test.png")

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.show()

def plot_value_array(i, predictions_array):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')

predictions_single = probability_model.predict(img)

print("J'ai predit :")
print(class_names[np.argmax(predictions_single)])

print(predictions_single)
plot_value_array(1, predictions_single[0])
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()