import os, cv2, re
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd

image_width = 150
image_height = 150
TRAIN_DIR_PATH = './inputs/train/'
TEST_DIR_PATH = './inputs/test/'
train_images = [TRAIN_DIR_PATH+i for i in os.listdir(TRAIN_DIR_PATH)]
test_images = [TEST_DIR_PATH+i for i in os.listdir(TEST_DIR_PATH)]




# funkcje pomocnicze do sortowania obrazkow
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]



# pobieram 1300 obrazow, zeby przyspieszyc proces
train_images.sort(key=natural_keys)
train_images = train_images[0:1300] + train_images[12500:13800] 
test_images.sort(key=natural_keys)


# resizing obrazkow i nadanie labeli : 0 - kot, 1 - pies
def prepare_data(list_of_images):
  x = [] # obrazy po resizingu
  y = [] # lista labeli
  for image in list_of_images:
      x.append(cv2.resize(cv2.imread(image), (image_width,image_height), interpolation=cv2.INTER_CUBIC))
  for i in list_of_images:
      if 'dog' in i:
          y.append(1)
      elif 'cat' in i:
          y.append(0)
  return x, y


# X - obrazy, Y - labele
X, Y = prepare_data(train_images)

# Dziele zbiór danych zawierający 2600 obrazów na 2 części - zbiór treningowy i zbiór testowy.
# 80% - dane treningowe, 20% - dane testowe
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16


# BUDOWANIE MODELU
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(image_width, image_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# generatory do zestawów treningowych i walidacyjnych
train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)


# trenowanie modelu
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

X_test, Y_test = prepare_data(test_images) 

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

counter = range(1, len(test_images) + 1)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)


# zapisuje rezultat do pliku
solution.to_csv("results.csv", index = False)

# Implementacja przedstawia rozwiazanie zadania klasyfikacji kotow i psow, gdzie dane to gotowe obrazy
# W pierwszym etapie należało przygotować dane tj. uspójnić wymiary obrazów do jednakowych rozmiarów oraz skategoryzować dane
# binarnie: 1 - pies, 0 - kot
# Do stworzenia sieci neuronowej użyłem modelu sequential z Kerasa. Model ten służy do konstruowania prostych modeli z liniowym stosem layerów.
# Następnie wytrenowałem go na podstawie podzielonych danych w proporcji train/test = 80/20.
# Liczba epok = 30 i steps_per_epoch = 130 dało zadowalający wynik accuracy wynoszący około 0.8
# Następnie wytrenowany model użyłem do predykcji danych testowych. Wstępna analiza potwierdziła poprawność działania modelu



