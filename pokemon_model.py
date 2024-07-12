import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from rich.progress import Progress, track
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
initial_epochs = input("How many epoch's should be completed? (Defaults to 100): ")
try:
    initial_epochs = int(initial_epochs)
except:
    print(f"[ERROR] Invalid Input - {initial_epochs}\nDefaulting to 100.")
    initial_epochs = 100
# Enable mixed precision

# Set the directory for the dataset
dataset_dir = "pokemon/images"

# Ensure the dataset path is correct
if not os.path.exists(dataset_dir):
    print(f"[ERROR] Dataset directory {dataset_dir} does not exist.")
    exit()
else:
    print(f"[INFO] Dataset directory {dataset_dir} found.")

# Function to load images and labels manually
def load_dataset(directory):
    images = []
    labels = []
    # Get class names
    class_names = sorted(os.listdir(directory))
    # Iterate through classes of images in the dataset
    for class_idx, class_name in track(enumerate(class_names), description=f"Loading Dataset at {directory}"):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                try:
                    # Loads image, preferably at 224x224px
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"[ERROR] Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels), class_names

# Load dataset
images, labels, class_names = load_dataset(dataset_dir)

# Duplicate samples of minority classes to ensure each class has at least two instances
label_counts = Counter(labels)
for label, count in label_counts.items():
    if count < 2:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        images = np.concatenate([images, images[indices]], axis=0)
        labels = np.concatenate([labels, labels[indices]], axis=0)

# Split dataset into training and validation sets with stratification
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.5, stratify=labels, random_state=123)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(class_names))
y_val = to_categorical(y_val, num_classes=len(class_names))

# Create tf.data datasets
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(64)
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(64)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Prepare the base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax', dtype='float32')(x)

# Model definition
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

# Progress callback using rich
class RichProgressCallback(Callback):
    def __init__(self, total_steps):
        super().__init__()
        self.progress = Progress()
        self.task = self.progress.add_task("Training", total=total_steps)
        self.progress.start()
        self.start_time = None
        self.epoch_durations = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        self.epoch_durations.append(duration)
        self.progress.update(self.task, advance=1)

    def on_train_end(self, logs=None):
        self.progress.stop()

# Training function with continuation option
def train_model(initial_epochs, continue_epochs=None):
    total_epochs = initial_epochs
    if continue_epochs:
        total_epochs += continue_epochs

    progress_callback = RichProgressCallback(total_steps=total_epochs)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        initial_epoch=initial_epochs - continue_epochs if continue_epochs else 0,
        callbacks=[checkpoint, early_stopping, progress_callback],
        verbose=0  # Suppress TensorFlow's progress bar
    )

    # Print the time taken for each epoch
    for epoch, duration in enumerate(progress_callback.epoch_durations):
        print(f"Epoch {epoch + 1}: {duration:.2f} seconds")

    # Load the best model
    model.load_weights('best_model.keras')

    # Evaluate the model
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return history

# Initial training
history = train_model(initial_epochs=initial_epochs)

# Tkinter GUI
class TrainingGUI(tk.Tk):
    def __init__(self, history):
        super().__init__()
        self.history = history
        self.title("Training Results")
        self.configure(bg='#333333')
        self.geometry("800x600")

        self.create_widgets()
        self.plot_results()

    def create_widgets(self):
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.add_epochs_button = ttk.Button(self.button_frame, text="Add Epochs", command=self.add_epochs)
        self.add_epochs_button.pack(side=tk.BOTTOM, pady=10)

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.legend()
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_epochs(self):
        additional_epochs = simpledialog.askinteger("Input", "How many more epochs would you like to add?", parent=self)
        if additional_epochs:
            new_history = train_model(initial_epochs=initial_epochs, continue_epochs=additional_epochs)
            self.history = new_history
            self.plot_results()

# Display the Tkinter GUI
app = TrainingGUI(history)
app.mainloop()

# Prediction function
def predict_pokemon(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    
    print(f"This image most likely belongs to {predicted_class} with a {100 * np.max(score):.2f} percent confidence.")
    
    # Return the image and prediction scores for later plotting
    return img, score.numpy()  # Convert tensor to numpy array

# Example prediction
img, score = predict_pokemon("test_images/deino.jpg")

# Display the image and prediction rankings in the same window
def display_results(img, score, class_names):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[np.argmax(score)]}")
    plt.axis('off')
    
    top_n = 5
    top_indices = np.argsort(score)[-top_n:][::-1]
    top_scores = score[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    plt.subplot(1, 2, 2)
    plt.barh(top_classes, top_scores, color='skyblue')
    plt.xlabel('Confidence')
    plt.title('Top Predictions')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# Display results
display_results(img, score, class_names)
