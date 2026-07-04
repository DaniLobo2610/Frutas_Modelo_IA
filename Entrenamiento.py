# ==================================================
# IMPORTACIONES
# ==================================================

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Importaciones cargadas correctamente.")

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 9
LEARNING_RATE = 0.0001
SEED = 42

# Rutas
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("Modelo_frutas")

MODEL_DIR.mkdir(exist_ok=True)

print("\nConfiguración cargada correctamente.")
print(f"Tamaño imagen: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Dataset: {DATASET_DIR}")
print(f"Modelo: {MODEL_DIR}")

# ==================================================
# DATASET
# ==================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    zoom_range=0.30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=(0.7, 1.3),
    channel_shift_range=20,
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR / "Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

validation_generator = test_datagen.flow_from_directory(
    DATASET_DIR / "Validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    DATASET_DIR / "Test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("\nDataset cargado correctamente.")

print(f"Clases detectadas: {train_generator.class_indices}")

print(f"Entrenamiento: {train_generator.samples}")

print(f"Validación: {validation_generator.samples}")

print(f"Prueba: {test_generator.samples}")

# ==================================================
# VISUALIZACIÓN DEL DATASET
# ==================================================

images, labels = next(train_generator)

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    img = (images[i] + 1) / 2
    img = np.clip(img, 0, 1)

    plt.imshow(img)

    plt.title(
        list(train_generator.class_indices.keys())[np.argmax(labels[i])]
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

print("\nVisualización completada.")


# ==================================================
# CREACIÓN DEL MODELO
# ==================================================

# Modelo base preentrenado en ImageNet
base_model = tf.keras.applications.MobileNetV2(

    input_shape=IMG_SIZE + (3,),

    include_top=False,

    weights="imagenet"

)

# Descongelar el modelo
base_model.trainable = True

# Congelar todas las capas excepto las últimas 30
for layer in base_model.layers[:-30]:

    layer.trainable = False


# Entrada del modelo
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

# Pasar la imagen por MobileNetV2
x = base_model(inputs, training=False)

# Reducir dimensiones
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Dropout
x = tf.keras.layers.Dropout(0.30)(x)

# Capa de salida
outputs = tf.keras.layers.Dense(

    train_generator.num_classes,

    activation="softmax"

)(x)

# Modelo final
model = tf.keras.Model(inputs, outputs)

print("\nModelo MobileNetV2 creado correctamente.")



# ==================================================
# COMPILACIÓN
# ==================================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=LEARNING_RATE

)

model.compile(

    optimizer=optimizer,

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModelo compilado correctamente.")


# ==================================================
# ENTRENAMIENTO
# ==================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        verbose=1

    )

]

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks

)

print("\nEntrenamiento finalizado correctamente.")


# ==================================================
# EVALUACIÓN
# ==================================================

loss, accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print("\n===================================")

print(f"Accuracy final: {accuracy*100:.2f}%")

print(f"Loss final: {loss:.4f}")

print("===================================")


# ==================================================
# GUARDAR MODELO
# ==================================================

model.save(

    MODEL_DIR / "modelo_frutas.keras"

)

model.save(

    MODEL_DIR / "modelo_frutas.h5"

)

class_names = list(

    train_generator.class_indices.keys()

)

with open(

    MODEL_DIR / "class_names.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        class_names,

        f,

        ensure_ascii=False,

        indent=4

    )

print("\n========================================")

print("Modelo guardado correctamente")

print("========================================")

print(f"Modelo : {MODEL_DIR/'modelo_frutas.keras'}")

print(f"Clases : {MODEL_DIR/'class_names.json'}")