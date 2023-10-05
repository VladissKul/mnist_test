import logging

from test1 import test_division

# получение пользовательского логгера и установка уровня логирования
py_logger = logging.getLogger(__name__)
py_logger.setLevel(logging.INFO)

# настройка обработчика и форматировщика в соответствии с нашими нуждами
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

# добавление форматировщика к обработчику
py_handler.setFormatter(py_formatter)
# добавление обработчика к логгеру
py_logger.addHandler(py_handler)

py_logger.info(f"Testing the custom logger for module {__name__}...")

x_vals = [2, 3, 6, 4, 10]
y_vals = [5, 7, 12, 0, 1]

for x_val, y_val in zip(x_vals, y_vals):
    x, y = x_val, y_val
    # вызов test_division
    test_division(x, y)
    py_logger.info(f"Call test_division with args {x} and {y}")
