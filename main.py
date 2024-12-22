import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

data_dir = 'dataset_clothes'

image_size = (224, 224)
batch_size = 32

datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

x = base_model.output
x = Dense(1024, activation='relu')(x) #  Добавляем полносвязный слой
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x) #  Выходной слой с количеством нейронов, равным количеству классов

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

for layer in model.layers[-5:]:
    layer.trainable = True
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

model.save('clothes.h5')

plt.plot(history.history['accuracy'] + history_fine_tune.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine_tune.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

classes_names = {0: 'dress', 1: 'hat', 2: 'longsleeve', 3: 'outwear', 4: 'pants', 5: 'shirt', 6: 'shoes', 7: 'shorts',
                 8: 'skirt', 9: 't-shirt'}

# img_path = 'images/shirt1.jpg'
# img = image.load_img(img_path, target_size=image_size)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# preds = model.predict(x)
# predicted_class_index = np.argmax(preds)
# print(f"Предсказанный класс: {classes_names.get(int(predicted_class_index))}")

test_dir = 'dataset_test'

y_true = []
y_pred = []

for class_name in classes_names.values():
    class_dir = os.path.join(test_dir, class_name)
    if os.path.exists(class_dir):
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            try:
                img = image.load_img(img_path, target_size=image_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                predicted_class_index = np.argmax(preds)
                predicted_class = classes_names.get(int(predicted_class_index))

                y_pred.append(predicted_class)
                true_class = class_name
                y_true.append(true_class)
            except Exception as e:
                print(f"Ошибка при обработке изображения {img_path}: {e}")

y_true_numeric = [list(classes_names.values()).index(label) for label in y_true]
y_pred_numeric = [list(classes_names.values()).index(label) for label in y_pred]

accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
print(f"Точность модели на тестовом множестве: {accuracy:.4f}")