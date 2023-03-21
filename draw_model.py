import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras import layers


dir = './dataset'
image_size = (100,100)
batch_size = 32

# Chargement du dataset d'entrainement
train_data = keras.preprocessing.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size
)

# Chargement du dataset de validation
valid_data = keras.utils.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size
)

plt.figure(figsize=(8, 8))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        data = images[i].numpy().astype("uint8")
        plt.imshow(data, cmap='gray', vmin=0, vmax=255)
        plt.title(train_data.class_names[labels[i]])
        plt.axis("off")

plt.show()

nbClasses = 345


model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(100, 100, 1)),
    layers.Conv2D(6, (3,3), activation='relu'),
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(12, (3,3), activation='relu'),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),

    layers.Dense(850, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.05),

    layers.Dense(600, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.05),

    layers.Dense(450, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.05),

    layers.Dense(nbClasses, activation='softmax'),
])

model.summary()

earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta =0, 
                                             patience=6, 
                                             verbose=1, 
                                             mode='min',
                                             restore_best_weights=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, validation_data=valid_data,
          epochs=30,
          callbacks=earlystopper)

(test_acc, test_loss) = model.evaluate(train_data,  valid_data, verbose=2)

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

model.save('draw.h5')