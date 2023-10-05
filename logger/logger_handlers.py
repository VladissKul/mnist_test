import logging


class LoggerHandlers:

    def __init__(self):
        pass

    @property
    def get_file_handler(self):
        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(self.config['log_format']))
        return file_handler

    @property
    def get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(self.config['log_format']))
        return stream_handler
