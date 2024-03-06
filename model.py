import os
import keras
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM,Reshape,LeakyReLU,BatchNormalization
from keras.optimizers import Adam,Adagrad,Adamax,Adadelta,AdamW
from keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import classification_report

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)
print(keras.__version__)
BS = 256
TS = (24, 24)
train_batch = generator('newdataset/train', shuffle=True, batch_size=BS, target_size=TS, class_mode='categorical')
valid_batch = generator('newdataset/test', shuffle=True, batch_size=BS, target_size=TS, class_mode='categorical')

SPE = len(train_batch)
VS = len(valid_batch)

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.5),
    Reshape((1, 512)),  # Reshape to add the third dimension (timesteps)
    LSTM(256),
    Dense(2, activation='softmax')
])

# Use Adam optimizer with a lower learning rate
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch, epochs=10, steps_per_epoch=SPE, validation_steps=VS)

# Generate predictions for the validation set
valid_preds = model.predict(valid_batch)
valid_labels = to_categorical(valid_batch.classes)

# Convert predictions to class labels
valid_pred_labels = valid_preds.argmax(axis=1)
valid_true_labels = valid_labels.argmax(axis=1)

# Display classification report
print("Classification Report:")
print(classification_report(valid_true_labels, valid_pred_labels))

model.save('models/cnnCat.h5', overwrite=True)
