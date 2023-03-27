import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras import layers
from keras.preprocessing.image import ImageDataGenerator



dir = './dataset'
image_size = (28,28)
batch_size = 64

class_names=[]
with open("./classes.txt", 'r') as f:
    class_names = [line.rstrip('\n') for line in f]

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

valid_data = keras.utils.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size
)

# train_datagen = ImageDataGenerator(
#     validation_split=0.2,
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.25,
#     height_shift_range=0.25,
#     shear_range=0.10,
#     zoom_range=[0.5,1.5])


# valid_datagen = ImageDataGenerator(validation_split=0.2,
#                                    rescale=1./255)


# plt.figure(figsize=(8, 8))
# for images, labels in train_data.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         data = images[i].numpy().astype("uint8")
#         plt.imshow(data, cmap='gray', vmin=0, vmax=255)
#         plt.title(train_data.class_names[labels[i]])
#         plt.axis("off")

# plt.show()


# train_batches = train_datagen.flow_from_directory(
#     dir,
#     target_size=image_size,
#     subset="training",
#     color_mode="grayscale",
#     batch_size=batch_size,
#     seed=123,
#     class_mode="sparse",
#     shuffle=True,
#     classes=class_names
# )

# valid_batches = valid_datagen.flow_from_directory(
#     dir,
#     target_size=image_size,
#     subset="validation",
#     color_mode="grayscale",
#     batch_size=batch_size,
#     seed=123,
#     class_mode="sparse",
#     shuffle=True,
#     classes=class_names
# )

nbClasses = 345
input_shape = (28,28,1)


model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=input_shape),
    layers.BatchNormalization(),

    layers.Conv2D(6, (3,3), padding='same', activation='relu'),
    layers.Conv2D(8, (3,3), padding='same', activation='relu'),
    layers.Conv2D(10, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(12, (3,3), padding='same', activation='relu'),
    layers.Conv2D(14, (3,3), padding='same', activation='relu'),
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),

    layers.Dense(700, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(500, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(400, activation='relu'),
    layers.Dropout(0.2),

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

history = model.fit(train_data,
                    validation_data=valid_data, 
                    epochs=50,
                    verbose=1,
                    callbacks=[earlystopper])

# (test_acc, test_loss) = model.evaluate(train_data,  valid_data, verbose=2)

model.save('draw2.h5')

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

