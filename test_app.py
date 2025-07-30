import tensorflow as tf

# Load the existing model (SavedModel format folder)
model = tf.keras.models.load_model("waste_model")

# Save it as an HDF5 file (.h5)
model.save("waste_model.h5")

print("✅ Model converted to waste_model.h5 successfully!")
