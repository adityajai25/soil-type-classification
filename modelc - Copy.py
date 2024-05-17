from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set batch size
batch_size = 32
num_classes = 3  # Assuming binary classification, change accordingly for multiclass

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# Update the directory paths using double backslashes or a raw string
train_generator = train_datagen.flow_from_directory(
    r'D:\chest xray\train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    r'D:\chest xray\test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# AlexNet Model
model_alexnet = Sequential([
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((3, 3), strides=(2, 2)),
    BatchNormalization(),

    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    BatchNormalization(),

    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    BatchNormalization(),

    Flatten(),
    
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model_alexnet.summary()

model_alexnet.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(learning_rate=0.001),
                      metrics=['accuracy'])

# Train the AlexNet model
history_alexnet = model_alexnet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the AlexNet model
model_alexnet.save('alexnet_model.h5')

# Visualize training history
plt.plot(history_alexnet.history['accuracy'], label='Training Accuracy')
plt.plot(history_alexnet.history['loss'], label='Training Loss')
plt.plot(history_alexnet.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_alexnet.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
