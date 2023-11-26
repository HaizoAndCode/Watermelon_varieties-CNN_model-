import telebot
from PIL import Image
import numpy as np
from tensorflow import keras
import io  # импортировать модуль io для работы с потоками байтов

bot = telebot.TeleBot('token_your_bot')
model = keras.models.load_model('my_model.keras')

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    # Получаем информацию о изображении
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file = bot.download_file(file_info.file_path)

    # Преобразуем изображение в массив numpy
    image = Image.open(io.BytesIO(file))
    image = image.resize((128, 128))  # Здесь нужно убедиться, что изображение соответствует размеру, на котором обучалась модель
    image = np.array(image) / 255.0

    # Предсказываем класс изображения с помощью загруженной модели
    prediction = model.predict(np.array([image]))
    predicted_class = np.argmax(prediction)

    # Отправляем ответ пользователю с предсказанным классом
    bot.send_message(message.chat.id, f"This looks like class {predicted_class}")

bot.polling(none_stop=True)
