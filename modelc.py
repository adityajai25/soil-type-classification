from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set batch size
batch_size = 32
num_classes = 4  # Assuming you have 4 classes for soil types

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling only for test set
test_datagen = ImageDataGenerator(rescale=1/255)

# Train and test data generators
train_generator = train_datagen.flow_from_directory(
    r'/Users/adityajp/Desktop/soil type/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    r'/Users/adityajp/Desktop/soil type/test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define AlexNet model
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
    Dropout(0.5),  # Adding dropout layer to reduce overfitting
    Dense(4096, activation='relu'),
    Dropout(0.5),  # Adding dropout layer to reduce overfitting
    Dense(num_classes, activation='softmax')
])

model_alexnet.summary()

model_alexnet.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(learning_rate=0.001),
                      metrics=['accuracy'])

# Train the model
history_alexnet = model_alexnet.fit(
    train_generator,
    epochs=20,  # Increase the number of epochs for better convergence
    validation_data=test_generator
)

# Save the model
model_alexnet.save('alexnet_model.h5')

# Visualize training history
plt.plot(history_alexnet.history['accuracy'], label='Training Accuracy')
plt.plot(history_alexnet.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
