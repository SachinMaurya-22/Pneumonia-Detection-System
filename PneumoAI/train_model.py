
"""
PneumoAI: Pneumonia Classification Model Training

Dataset structure:
dataset/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

Output:
models/pneumonia_model.keras
models/class_names.json
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------- CONFIGURATION ----------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "pneumonia_model.keras"
CLASS_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(split, shuffle):
    """Load a dataset split with a fixed class order."""

    directory = DATA_DIR / split

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {directory}"
        )

    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    # Match the Flask preprocessing pipeline: pixel values [0, 1].
    dataset = dataset.map(
        lambda images, labels: (images / 255.0, labels),
        num_parallel_calls=AUTOTUNE,
    )

    return dataset.prefetch(AUTOTUNE)


def build_model():
    """Create a transfer-learning image classifier."""

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="chest_xray",
    )

    # Augmentation runs during training, not inference.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.03)(x)

    # Convert [0, 1] inputs to [-1, 1] for MobileNetV2.
    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_normalization",
    )(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    # Keep pretrained batch-normalization layers in inference mode.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="classification",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, backbone


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("train", shuffle=True)
    val_ds = load_dataset("val", shuffle=False)
    test_ds = load_dataset("test", shuffle=False)

    # Verify the directory ordering matches the model output.
    with open(CLASS_PATH, "w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, indent=4)

    model, backbone = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    print("\nStage 1: Training the classification head")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    print("\nStage 2: Fine-tuning selected backbone layers")

    backbone.trainable = True

    # Fine-tune only the final portion of the pretrained backbone.
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    # Do not update batch-normalization statistics on a small dataset.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    # Load the best checkpoint selected using validation loss.
    best_model = keras.models.load_model(MODEL_PATH, compile=False)

    print("\nFinal evaluation on the held-out test set")

    evaluation_model = keras.models.load_model(
        MODEL_PATH,
        compile=True,
    )

    evaluation_model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    results = evaluation_model.evaluate(
        test_ds,
        return_dict=True,
    )

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Ensure the final saved artifact exists and is loadable.
    best_model.save(MODEL_PATH)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Class labels saved: {CLASS_PATH}")
    print("\nVerify test performance before using the model.")


if __name__ == "__main__":
    main()
