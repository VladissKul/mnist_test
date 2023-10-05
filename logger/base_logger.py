import logging
import os

from logger.hdfs_storage import HdfsStorage
from .logger_handlers import LoggerHandlers
from .logger_messanges import LoggerMessanges


class BaseLogger(LoggerHandlers, LoggerMessanges):

    def __init__(self):
        self.hs = HdfsStorage()
        self.load_config()
        super(BaseLogger, self).__init__()
        self.create_logger()

    def load_config(self):
        self.config = {
            "log_file": "text.log",
            "log_format_json": """{"time":"%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}""",
            "log_format": """%(asctime)s %(levelname)-8s %(name)s: %(message)s""",
            "level": 10,
            "local_logs_folder": f"""{os.getcwd()}/logs"""
        }

    # def create_logger(self):
    #     self.logger = logging.getLogger('logger')
    #     self.logger.setLevel(logging.INFO)
    #     self.logger.addHandler(self.get_file_handler)  # Добавить обработчик для записи в файл
    #     self.logger.addHandler(self.get_stream_handler)  # Добавить обработчик для вывода в консоль

    def create_logger(self):
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)

        # Обработчик для записи в файл logs.log
        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(self.config['log_format']))

        self.logger.addHandler(file_handler)

    def save_logs_to_hdfs(self):
        self.hs.upload_file(os.environ['logs_local_fileways'], os.environ['logs_hdfs_fileways'])
