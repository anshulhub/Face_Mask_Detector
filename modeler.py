import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2

width = 224
height = 224

batch_size = 32
data_dir = r"C:\Users\anshu\Desktop\Face Mask Detection\dataset"

training = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)
classes = training.class_names


for images, labels in training.take(1):
    plt.imshow(images[1].numpy().astype('uint8'))
    plt.title(classes[labels[1]])

model = MobileNetV2(weights='imagenet')

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
            
# model.summary()
face_mask_detection = model.fit(training,validation_data=validation,epochs=3)


img = tf.keras.preprocessing.image.load_img('', target_size=(height, width))

image_array = tf.keras.preprocessing.image.img_to_array(img)

image_array = tf.expand_dims(image_array,0)

image_array.shape

predictions = model.predict(image_array)

score = tf.nn.softmax(predictions[0])

import numpy
print(classes[np.argmax(score)], 100*np.max(score))

model.save('detection.model', save_format="h5")

acc = face_mask_detection.history['accuracy']
val_acc = face_mask_detection.history['val_accuracy']

loss= face_mask_detection.history['loss']
val_loss= face_mask_detection.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()