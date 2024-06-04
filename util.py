import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, callbacks
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, Conv2D, MaxPooling2D, Reshape, LSTM, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
from tensorflow.keras import regularizers
from art.attacks.evasion import FastGradientMethod
from tensorflow.keras.models import load_model

path="/content/drive/MyDrive/AttackOnMLModel/"
path='.'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
CLASSES = ["COVID", "non-COVID"]
NUM_CLASSES = len(CLASSES)


def load_model_(str_):
  return load_model(path+f'/Models/CNN-LSTM-binary_crossentropy-{str_}.keras')

model = load_model_('original-reshape-2')


def generate_image_csv(root_dir, output_csv):
    # Inizializza le liste per i percorsi delle immagini e le classi
    image_paths = []
    image_classes = []

    # Itera su ogni directory nel directory radice
    for dir_name in os.listdir(root_dir):
        # Il nome della directory è la classe dell'immagine
        image_class = dir_name

        # Il percorso della directory
        dir_path = os.path.join(root_dir, dir_name)

        # Se il percorso non è una directory, salta al prossimo
        if not os.path.isdir(dir_path):
            continue

        # Itera su ogni file nella directory
        for file_name in os.listdir(dir_path):
            # Il percorso del file
            file_path = os.path.join(dir_path, file_name)

            # Se il file non è un'immagine, salta al prossimo
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Aggiungi il percorso del file e la classe dell'immagine alle liste
            image_paths.append(file_path)
            image_classes.append(image_class)

    # Crea un DataFrame con i percorsi delle immagini e le classi
    df = pd.DataFrame({
        'image': image_paths,
        'label': image_classes
    })

    # Scrivi il DataFrame in un file CSV
    df.to_csv(output_csv, index=False)


def fit_model(model, str_:str, X_train, y_train, patience_to_stop=2):
  # Define callbacks
  callbacks_list = [
    callbacks.EarlyStopping(monitor='accuracy', patience=patience_to_stop),  # Stop training if accuracy doesn't improve after [patience_to_stop] epochs
      callbacks.ModelCheckpoint(filepath=path+f'/Models/CNN-LSTM-binary_crossentropy-{str_}.keras', monitor='loss'),  # Save the best model based on loss
  ]

  # Fit the model
  history = model.fit(X_train, y_train, batch_size=16, epochs=25, callbacks=callbacks_list, validation_split=0.2, verbose=True)

  model.save(path+f'/Models/CNN-LSTM-binary_crossentropy-{str_}.keras')
  model.save(path+f'/Models/CNN-LSTM-binary_crossentropy-{str_}.h5')

  return model, history


def print_history(history):
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Validation')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Train')
  plt.plot(history.history['val_accuracy'], label='Validation')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()


def perform_prediction(model, X_test):
  y_pred = model.predict(X_test, batch_size=16)
  return y_pred


def print_confusion_matrix(y_pred, y_test):
  from sklearn.metrics import confusion_matrix, classification_report
  import seaborn as sns
  import numpy as np

  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true_classes = np.argmax(y_test, axis=1)

  cm = confusion_matrix(y_true_classes, y_pred_classes)
  plt.figure(figsize=(5, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0: COVID', '1: non-COVID'],
              yticklabels=['0: COVID', '1: non-COVID'])
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()

  print("Classification Report:")
  print(classification_report(y_true_classes, y_pred_classes, zero_division=0.0))


def print_model_evaluation(model, X_test, y_test):
  score, acc, p, r = model.evaluate(X_test, y_test, batch_size=16)
  print('Test Loss =', score)
  print("Test accuracy: %.2f%%" % (acc * 100))


def print_image_datasets(X_data, y_data):
  fig, axes = plt.subplots(5, 5, figsize=(7, 7))

  for i, ax in enumerate(axes.flat):
      ax.imshow(X_data[i, ...].squeeze(), cmap='gray')
      ax.axis("off")
      ax.text(
          0.5,
          -0.05,
          f"{CLASSES[np.argmax(y_data, axis=1)[i]]}",
          transform=ax.transAxes,
          horizontalalignment="center",
          verticalalignment="center",
          fontsize=10,
      )

  plt.tight_layout()
  plt.suptitle("Datasets", fontsize=16, y=1.01)
  plt.show()

def get_compile_model():
    model_ = create_my_model() #learning_rate=0.0001
    model_.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model_


def create_my_model():
  model = models.Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='sigmoid'))

  return model

