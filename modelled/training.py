# Import modules
import datetime
import math
import multiprocessing

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Read dataset
DATASET_PATH = "./dataset/fer2013.csv"
df = pd.read_csv(DATASET_PATH)

# Info
print("Shape of images: {length}x{length}".format(length=int(math.sqrt(len(df.pixels[35886].split(' '))))))
emotion_label = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

# Reshape pixels column to make an array of shape (35887, 48, 48)
img_array = df['pixels'].apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('float32'))
img_array = np.stack(img_array, axis=0)

# Add grayscale
img_features = []
for i in range(len(img_array)):
    temp = cv2.cvtColor(img_array[i], cv2.COLOR_GRAY2RGB)
    img_features.append(temp)

img_features = np.array(img_features)

# Encode image labels
le = LabelEncoder()
img_labels = le.fit_transform(df['emotion'])
img_labels = np_utils.to_categorical(img_labels)


# Split dataset into test, training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    img_features, img_labels,
    stratify=img_labels,
    test_size=0.1,
    random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    stratify=y_train,
    test_size=0.2,
    random_state=42
)

# Normalizing pixel values
X_train = X_train / 255.
X_test = X_test / 255.
X_valid = X_valid / 255.

IMG_WIDTH = 48
IMG_HEIGHT = 48
CHANNELS = 3

# Download and/or Load VGGNet-19, a 19-layer CNN
vgg = tf.keras.applications.VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)
)
print("\n\nVGG19 Model Summary:")
vgg.summary()


# Customizing the VGG19 Model
# Adding one GlobalAvgPooling layer instead of MaxPooling2D
# Adding Dense layer according the `num_classes` present
def get_model(build, classes):
    _model = build.layers[-2].output
    _model = GlobalAveragePooling2D()(_model)
    _model = Dense(classes, activation='softmax', name='output_layer')(_model)

    return _model


num_classes = 7
head = get_model(vgg, num_classes)
model = Model(inputs=vgg.input, outputs=head)
print("\n\nBuild Model Summary:")
model.summary()

# Stop training when parameter updates don't improve on a
# validation set (poor scoring metric scores)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True
)

# Reduce learning rate when accuracy has stopped improving
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)
callbacks = [early_stopping, lr_scheduler]

# Generate new variations of the images at each epoch
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
train_datagen.fit(X_train)

# Compile model and check for formatting errors, define the
# loss function, the optimizer, learning rate, and the
# metrics (accuracy).
batch_size = 256
epochs = 15
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999
    ),
    metrics=['accuracy']
)


# Training the model
history = model.fit(
    train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size
    ),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)


# Accuracy and Loss graph plot
sns.set()
fig = pyplot.figure(0, (12, 4))

ax1 = pyplot.subplot(1, 2, 1)
sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid')
pyplot.title('Accuracy')
pyplot.tight_layout()

ax2 = pyplot.subplot(1, 2, 2)
sns.lineplot(x=history.epoch, y=history.history['loss'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid')
pyplot.title('Loss')
pyplot.tight_layout()

pyplot.savefig('epoch_history_dcnn.png')

# Saving the model
model.save(
    './weights/emotion-vgg19-{timestamp}.h5'
    .format(timestamp=datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
)
