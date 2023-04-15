import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import cv2


numbers = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = numbers.load_data()

train_images = train_images.reshape(60000,28,28,1).astype('float32')
test_images = test_images.reshape(10000,28,28,1).astype('float32')

print(train_labels)

class_names = ['Zero', 'Un', 'Deux', 'Trois', 'Quatre', 'Cinq', 'Six', 'Sept', 'Huit', 'Neuf']

print(test_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.25,
                             height_shift_range=0.25,
                             shear_range=0.10,
                             zoom_range=[0.5,1.5])

datagen.fit(train_images.reshape(train_images.shape[0], 28, 28, 1))


#Creation du model

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (5,5), activation='relu'),
    layers.Conv2D(32, (5,5), activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.05),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.05),
    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.05),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.05),

    layers.Dense(84, activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.05),

    layers.Dense(10, activation='softmax')
])

earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta =0, 
                                             patience=6, 
                                             verbose=1, 
                                             mode='min',
                                             restore_best_weights=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(datagen.flow(train_images,train_labels , batch_size=84),
                    validation_data=(test_images,test_labels),
                    epochs=30,
                    steps_per_epoch=train_images.shape[0] // 84,
                    callbacks=earlystopper)

(test_acc, test_loss) = model.evaluate(test_images,  test_labels, verbose=2)

print(history.history.keys())


plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('model_numbers.h5')

# print('\nTest accuracy:', test_acc)

# Pour tester les images du dataset de test
# predictions = model.predict(test_images)

# La premiere prediction
# print(predictions[0])

# print(np.argmax(predictions[0]))
# print(test_labels[0])
# plt.figure()
# plt.imshow(test_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



img = cv2.imread("lien_vers_l_image", 0)
img = cv2.bitwise_not(img)
img = cv2.resize(img, (28, 28))
img = np.expand_dims(img,0)


plt.figure()
plt.imshow(img[0])
plt.show()

predictions_single = model.predict(img)

print("J'ai predit :")
print(class_names[np.argmax(predictions_single)])


print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()