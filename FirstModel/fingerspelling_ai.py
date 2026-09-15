import tensorflow as tf
from tensorflow.keras import layers, models

train_dir = r"dataset/train"
# test_dir = r"dataset/test"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([

    data_augmentation,

    layers.Rescaling(1. / 255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.5),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(7, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.build((None, 64, 64, 3))
model.summary()

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

model.save("fingerspelling_model.keras")