import json
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, f1_score


# =========================
# GLOBAL SETTINGS
# =========================
SEED = 42
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 8
FINE_TUNE_LAST_N = 40

USE_ONLINE_AUG = False  # keep False because train set is already augmented in preprocessing


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# =========================
# PATHS
# =========================
def setup_paths():
    project_root = Path(__file__).resolve().parent

    split_data_path = project_root / "split_dataset_gray_balanced"

    models_path = project_root / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    results_path = project_root / "results" / "improved_gray_balanced"
    results_path.mkdir(parents=True, exist_ok=True)

    print("=== Path Configuration ===")
    print(f"Project root: {project_root}")
    print(f"Data path: {split_data_path}")
    print(f"Models path: {models_path}")
    print(f"Results path: {results_path}")
    print(f"Train exists: {(split_data_path / 'train').exists()}")
    print(f"Validation exists: {(split_data_path / 'validation').exists()}")
    print(f"Test exists: {(split_data_path / 'test').exists()}")

    return split_data_path, models_path, results_path


# =========================
# MODEL
# =========================
def create_model(input_shape, num_classes, alpha=0.75):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        alpha=alpha,
    )

    # Stage 1: freeze all backbone
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation=tf.nn.relu6)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# =========================
# PLOTS
# =========================
def save_training_plots(history_list, out_dir: Path):
    plt.figure(figsize=(6, 4))
    for name, h in history_list:
        plt.plot(h.history["accuracy"], label=f"{name} Train Acc")
        plt.plot(h.history["val_accuracy"], label=f"{name} Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for name, h in history_list:
        plt.plot(h.history["loss"], label=f"{name} Train Loss")
        plt.plot(h.history["val_loss"], label=f"{name} Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=200)
    plt.close()


def save_confusion_matrix(cm, class_names, out_dir: Path):
    np.save(out_dir / "confusion_matrix.npy", cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()


# =========================
# HELPERS
# =========================
def get_class_names_in_index_order(class_indices: dict):
    class_names = [None] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name
    return class_names


def create_generators(split_data_path: Path):
    if USE_ONLINE_AUG:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        split_data_path / "train",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb",  # MobileNetV2 needs 3 channels
    )

    val_gen = eval_datagen.flow_from_directory(
        split_data_path / "validation",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )

    test_gen = eval_datagen.flow_from_directory(
        split_data_path / "test",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )

    return train_gen, val_gen, test_gen


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    split_data_path, models_path, results_path = setup_paths()

    print("\n=== TensorFlow Info ===")
    print("TF =", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    train_gen, val_gen, test_gen = create_generators(split_data_path)

    class_names = get_class_names_in_index_order(train_gen.class_indices)
    print("\nClass indices:", train_gen.class_indices)
    print("Class names :", class_names)

    model, base_model = create_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=len(class_names),
        alpha=0.75,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    best_path = models_path / f"improved_gray_balanced_best_{run_id}.keras"
    final_path = models_path / f"improved_gray_balanced_final_{run_id}.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    # =========================
    # Stage 1: Feature Extraction
    # =========================
    print("\n=== Stage 1: Feature Extraction (Frozen Backbone) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h1 = model.fit(
        train_gen,
        epochs=EPOCHS_STAGE1,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================
    # Stage 2: Fine-Tuning
    # =========================
    print("\n=== Stage 2: Fine-Tuning (Unfreeze last N layers, freeze BN) ===")
    base_model.trainable = True

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    for layer in base_model.layers[:-FINE_TUNE_LAST_N]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen,
        epochs=EPOCHS_STAGE2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_path)
    save_training_plots([("stage1", h1), ("stage2", h2)], results_path)

    # =========================
    # Evaluation
    # =========================
    best_model = tf.keras.models.load_model(best_path)

    print("\n=== Test Evaluation (BEST) ===")
    test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)

    test_gen.reset()
    y_prob = best_model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    (results_path / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(cm, class_names, results_path)

    metrics = {
        "model": "MobileNetV2 (alpha=0.75 + ReLU6 head) on grayscale balanced dataset (2-stage fine-tuning)",
        "tf_version": tf.__version__,
        "dataset_path": str(split_data_path),
        "train_images": int(train_gen.samples),
        "val_images": int(val_gen.samples),
        "test_images": int(test_gen.samples),
        "class_names": class_names,
        "class_indices": train_gen.class_indices,
        "image_size": [IMG_HEIGHT, IMG_WIDTH],
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "stage1_epochs": EPOCHS_STAGE1,
        "stage2_epochs": EPOCHS_STAGE2,
        "stage1_lr": 1e-3,
        "stage2_lr": 1e-5,
        "fine_tune_last_N_layers": int(FINE_TUNE_LAST_N),
        "use_online_augmentation": bool(USE_ONLINE_AUG),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "best_model_path": str(best_path),
        "final_model_path": str(final_path),
        "run_id": run_id,
    }

    with open(results_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Saved Files ===")
    print("Best model :", best_path)
    print("Final model:", final_path)
    print("Results dir:", results_path)
    print(f"\nTest Acc: {test_acc:.4f} | Macro-F1: {macro_f1:.4f} | MCC: {mcc:.4f} | Kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()