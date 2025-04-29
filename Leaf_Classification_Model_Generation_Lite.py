import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import seaborn as sns
import subprocess

# Parameters
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = (96, 96)
LEARNING_RATE = 0.001

# Paths for saving models
model_dir = "/home/priya/mint_project/Freshness_Classification/model"
tflite_model_path = os.path.join(model_dir, "leaf_classifier_Lite_2.tflite")
cc_model_path = os.path.join(model_dir, "freshness_model.cc")

# Ensure the /model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to load and preprocess images
def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)  # Resize to a fixed size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to load dataset
def load_dataset(image_dir, label, img_size=IMG_SIZE):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = load_and_preprocess_image(image_path, img_size)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Function to split the dataset for each class equally
def split_data_equally(images, labels, test_size=0.2, val_size=0.5, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Load datasets for fresh, dried, and spoiled categories separately
fresh_images, fresh_labels = load_dataset(r"/home/priya/mint_project/Freshness_Classification/dataset_2/fresh", label='fresh')
dried_images, dried_labels = load_dataset(r"/home/priya/mint_project/Freshness_Classification/dataset_2/dried", label='dried')
spoiled_images, spoiled_labels = load_dataset(r"/home/priya/mint_project/Freshness_Classification/dataset_2/spoiled", label='spoiled')
unknown_images, unknown_labels = load_dataset(r"/home/priya/mint_project/Freshness_Classification/dataset_2/unknown", label='unknown')


# Combine all labels and fit LabelEncoder on all of them
all_labels = np.concatenate((fresh_labels, dried_labels, spoiled_labels, unknown_labels), axis=0)

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform the individual labels using the fitted LabelEncoder
fresh_labels = label_encoder.transform(fresh_labels)
dried_labels = label_encoder.transform(dried_labels)
spoiled_labels = label_encoder.transform(spoiled_labels)
unknown_labels = label_encoder.transform(unknown_labels)

# Split data equally for each class
X_train_fresh, X_val_fresh, X_test_fresh, y_train_fresh, y_val_fresh, y_test_fresh = split_data_equally(fresh_images, fresh_labels)
X_train_dried, X_val_dried, X_test_dried, y_train_dried, y_val_dried, y_test_dried = split_data_equally(dried_images, dried_labels)
X_train_spoiled, X_val_spoiled, X_test_spoiled, y_train_spoiled, y_val_spoiled, y_test_spoiled = split_data_equally(spoiled_images, spoiled_labels)
X_train_unknown, X_val_unknown, X_test_unknown, y_train_unknown, y_val_unknown, y_test_unknown = split_data_equally(unknown_images, unknown_labels)

# Combine the train, validation, and test sets for all labels
X_train = np.concatenate((X_train_fresh, X_train_dried, X_train_spoiled, X_train_unknown), axis=0)
y_train = np.concatenate((y_train_fresh, y_train_dried, y_train_spoiled, y_train_unknown), axis=0)

X_val = np.concatenate((X_val_fresh, X_val_dried, X_val_spoiled, X_val_unknown), axis=0)
y_val = np.concatenate((y_val_fresh, y_val_dried, y_val_spoiled, y_val_unknown), axis=0)

X_test = np.concatenate((X_test_fresh, X_test_dried, X_test_spoiled, X_test_unknown), axis=0)
y_test = np.concatenate((y_test_fresh, y_test_dried, y_test_spoiled, y_test_unknown), axis=0)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# # Function to plot images in a grid
# def plot_augmented_images(datagen, images, labels, label_names, num_examples=1):
#     plt.figure(figsize=(12, 12))
#     # Iterate over each class (fresh, dried, spoiled, unknown)
#     for i in range(4):
#         # Pick random index from each class images
#         random_idx = random.randint(0, len(images[i]) - 1)
#         # Generate augmented images from the randomly chosen image
#         augmented_img = datagen.flow(np.expand_dims(images[i][random_idx], axis=0), batch_size=1)

#         # Plot the original and first augmented image for each category
#         for j in range(num_examples):
#             ax = plt.subplot(4, num_examples, i * num_examples + j + 1)
#             img = next(augmented_img)[0]  # Get the first augmented image

#             plt.imshow(img)
#             plt.title(f'{label_names[i]}')
#             plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# # List of images and corresponding label names for visualization
# image_samples = [fresh_images, dried_images, spoiled_images, unknown_images]
# label_names = ['Fresh', 'Dried', 'Spoiled', 'Unknown']

# # Plot 1 augmented image per category
# plot_augmented_images(datagen, image_samples, [y_train_fresh, y_train_dried, y_train_spoiled, y_train_unknown], label_names, num_examples=1)


# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')  # 3 output units for the 3 leaf classes + 1 unknown class
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model with validation data
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save(os.path.join(model_dir, "leaf_classifier_Lite_2.h5"))

# Confusion matrix and plots
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_test

# Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compute accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# # Function to make a prediction using the TensorFlow Lite model (adjusted for INT8 input)
# def make_sample_prediction(tflite_model_path, test_image):
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()

#     # Get input and output tensors information
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Preprocess the test image for INT8 quantized model
#     # Convert the test image back to range [0, 255] and cast it to int8
#     test_image = (test_image * 255).astype(np.int8)
#     test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

#     # Set the input tensor
#     interpreter.set_tensor(input_details[0]['index'], test_image)

#     # Run inference
#     interpreter.invoke()

#     # Get the output tensor (predictions)
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     return output_data

# # Function to visualize the prediction result
# def visualize_prediction(test_image, predictions, label_names):
#     # Plot the input image
#     plt.figure(figsize=(6, 4))
#     plt.imshow(test_image)
#     plt.axis('off')
    
#     # Get the predicted class and the prediction scores
#     predicted_class = np.argmax(predictions)
#     predicted_label = label_names[predicted_class]
#     prediction_scores = predictions[0]

#     # Print prediction details
#     print(f"Predicted Class: {predicted_label}")
#     print(f"Prediction Scores: {prediction_scores}")

#     # Show prediction in title
#     plt.title(f'Predicted: {predicted_label}')
#     plt.show()

# # Sample prediction using TFLite model
# random_index = random.randint(0, len(X_test) - 1)  # Pick a random test image
# sample_test_image = X_test[random_index]  # Select the test image
# true_label = label_encoder.inverse_transform([y_test[random_index]])[0]  # Get the true label

# # Make prediction using TFLite model
# predictions = make_sample_prediction(tflite_model_path, sample_test_image)

# # Visualize the prediction result
# print(f"True Label: {true_label}")
# visualize_prediction(sample_test_image, predictions, label_encoder.classes_)


#####################
# Additional Sections for Conversion to TensorFlow Lite and C file
#####################

# Create a representative dataset for quantization
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32)).batch(1).take(100):
        yield [input_value]

# Convert the model to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Quantize inputs to int8
converter.inference_output_type = tf.int8  # Quantize outputs to int8
# ** Disable per-channel quantization for fully connected layers **
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_model = converter.convert()

# Save the TFLite model inside /model folder
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved at {tflite_model_path}")

# Convert .tflite file to .cc file using xxd, ensuring it doesn't include the full file path in the variable name
subprocess.run(["xxd", "-i", tflite_model_path, "model_temp.cc"])

# Read the generated file and modify it
with open("model_temp.cc", "r") as infile, open(cc_model_path, "w") as outfile:
    outfile.write('#include "freshness_model_data.h"\n\n')
    outfile.write('// Keep model aligned to 8 bytes to guarantee aligned 64-bit accesses.\n')
    outfile.write('alignas(8) const unsigned char leaf_classifier_model[] = {\n')

    # Skip the first line from xxd output
    infile.readline()

    # Process the content, skipping the file length definition initially
    for line in infile:
        if line.startswith('unsigned int'):
            break
        outfile.write(line)

    # Replace the length definition with a simplified variable name
    model_len_line = line.replace('unsigned int _home_priya_mint_project_Freshness_Classification_model_leaf_classifier_Lite_2_tflite_len =', 'const int leaf_classifier_len =')
    outfile.write('\n\n')
    outfile.write(model_len_line)

# Remove the temporary file generated by xxd
os.remove("model_temp.cc")

print(f"C file generated and saved at {cc_model_path}")

