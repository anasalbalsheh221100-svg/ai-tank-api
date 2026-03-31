import tensorflow as tf

model = tf.keras.models.load_model(
    "models/improved_gray_balanced_best_20260104_015524.keras"
)

model.save("models/model_fixed.h5")

print("DONE")