from logger.custom_logger import CustomLogger

logger = CustomLogger()


@logger.error
def logger_test():
    logger.error('Prediction:')
    print(1 + 2)


logger_test()