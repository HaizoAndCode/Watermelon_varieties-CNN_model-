from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image
import numpy as np

data_dir = r"Путь_к_data"

# Получаем список подпапок (классов)
classes = os.listdir(data_dir)

all_images = []
all_labels = []

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):  # Убедимся, что это действительно папка 
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if os.path.isfile(img_path):  # Убедимся, что это файл изображения 
                img = Image.open(img_path)
                # Предварительная обработка (масштабирование, изменение размера и т.д.), если необходимо 
                all_images.append(np.array(img))
                all_labels.append(class_name)

# Изменение размеров всех изображений до ожидаемых размеров (128x128)
resized_images = [np.array(Image.fromarray(image).resize((128, 128))) for image in all_images]

# Преобразуем изображения и метки в numpy массивы для удобства работы с библиотеками глубокого обучения 
X = np.array(resized_images)
y = np.array(all_labels)

# Преобразование меток классов из строк в числовой формат (предполагая, что каждый класс имеет свою числовую интерпретацию)
from sklearn.preprocessing import LabelEncoder

# Разделение загруженных данных на обучающий и тестовый наборы 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование значений пикселей в диапазон от 0 до 1
X_train = X_train.astype('float32') / 255.0  # Масштабирование до диапазона [0, 1]
X_test = X_test.astype('float32') / 255.0  # Масштабирование до диапазона [0, 1]

# Преобразование меток классов из строк в числовой формат (предполагая, что каждый класс имеет свою числовую интерпретацию)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Определение параметров изображения (размеры изображений были изменены до 128x128)
image_height = 128
image_width = 128
num_channels = 3  # для цветных изображений RGB

# Определение числа классов
num_classes = len(np.unique(y))  # количество уникальных меток в y

# Создаем модель сверточной нейронной сети 
model = models.Sequential([
    # Слой свертки с 32 небольшими фильтрами (3x3) и функцией активации ReLU 
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    # Слой пулинга для уменьшения размерности признаков 
    layers.MaxPooling2D((2, 2)),
    # Дополнительные слои свертки и пулинга могут быть добавлены по необходимости 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Преобразуем трехмерный выход пулинговых слоев в одномерный для подачи на вход полносвязной сети 
    layers.Flatten(),
    # Полносвязный слой с 128 нейронами и функцией активации ReLU 
    layers.Dense(128, activation='relu'),
    # Выходной слой с количеством нейронов, соответствующем количеству классов и функцией активации softmax 
    layers.Dense(num_classes, activation='softmax')
])

# Компиляция модели 
model.compile(optimizer='adam',  # Оптимизатор 
              loss='sparse_categorical_crossentropy',  # Функция потерь (для целых чисел - sparse_categorical_crossentropy) 
              metrics=['accuracy']  # Метрика оценки производительности модели 
             )

# Выводим сводку модели, чтобы убедиться, что все слои настроены правильно 
model.summary()

# Обучение модели 
history = model.fit(
    X_train, y_train,
    epochs=130,  # Количество эпох
    batch_size=32,  # Размер пакета
    validation_data=(X_test, y_test)  # Данные для валидации
)

# Оценка модели 
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Предположим, что ваша обученная модель называется model
model.save('my_model.keras')  # Сохранение модели в формате HDF5
