from logger.custom_logger import CustomLogger

logger = CustomLogger()


@logger.info
def func1(a, b):
    return a / b


print(func1(1, 2))
print(func1(1, 0))
