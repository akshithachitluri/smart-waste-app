import tensorflow as tf

# Load the SavedModel (you already have this folder)
model = tf.keras.models.load_model("waste_model", compile=False)

# Save it in the new Keras 3 format
model.save("waste_model.keras")
