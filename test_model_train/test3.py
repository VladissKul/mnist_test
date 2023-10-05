import cv2
import numpy as np


def preprocess_image(image_path):
    # Попытка открыть изображение с помощью OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or image.size == 0:
        # Если изображение не было успешно открыто или оно пусто, вернуть None
        return None

    # Изменяем размер изображения на 28x28 пикселей (стандартный размер MNIST)
    image = cv2.resize(image, (28, 28))

    # Инвертируем цвета (черное на белом вместо белого на черном)
    image = 255 - image

    # Бинаризуем изображение, преобразуя все пиксели в черный (0) или белый (1)
    threshold = 128  # Задаем порог бинаризации
    image = (image > threshold).astype(np.uint8)

    # Нормализуем значения пикселей к диапазону [0, 1]
    image = image.astype(np.float32) / 255.0

    # Разворачиваем изображение в одномерный массив размером 28x28=784
    image = image.reshape(784)

    return image


# Пример использования функции preprocess_image
image_path = r'/images/test.png'  # Замените на путь к вашему изображению
mnist_data = preprocess_image(image_path)

if mnist_data is not None:
    # Вы можете продолжить обработку данных, если изображение успешно преобразовано
    pass
else:
    print("Ошибка: Не удалось загрузить или обработать изображение.")
