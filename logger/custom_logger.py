import functools
import logging
import os
import time
import traceback

from logger.base_logger import BaseLogger
# from core.custom_messangers import CustomMessangers
from logger.base_messangers import BaseMessangers
from logger.hdfs_storage import HdfsStorage


class CustomMessangers(BaseMessangers):

    def __init__(self):
        super(CustomMessangers, self).__init__()

    def send_messange(self, email_text):
        self.send_email(email_text)


class CustomLogger(BaseLogger):

    def __init__(self):
        self.cm = CustomMessangers()
        self.hs = HdfsStorage()
        super(CustomLogger, self).__init__()

    def info(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            self.logger.debug('{} start'.format(func.__name__))
            try:
                result = func(*args, **kwargs)
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.info('{} done. Time: {} s'.format(func.__name__, elapsed_time))
                return result
            except Exception as e:
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.error('{} error. Time: {} s'.format(func.__name__, elapsed_time), exc_info=True)
            self.logger.debug('{} end'.format(func.__name__))

        return wrapper

    def debug(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            self.logger.debug('{} start'.format(func.__name__))
            try:
                result = func(*args, **kwargs)
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.debug('{} done. Time: {} s'.format(func.__name__, elapsed_time))
                return result
            except Exception as e:
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.error('{} error. Time: {} s'.format(func.__name__, elapsed_time), exc_info=True)
            self.logger.debug('{} end'.format(func.__name__))

        return wrapper

    def error(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.debug('{} start.'.format(func.__name__))
            try:
                t_start = time.time()
                result = func(*args, **kwargs)
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.info('{} done. Time: {} s'.format(func.__name__, elapsed_time))
                return result
            except Exception as e:
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.error('{} error'.format(func.__name__), exc_info=True)
            self.logger.debug('{} end'.format(func.__name__))

        return wrapper

    def warning(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            self.logger.debug('{} start'.format(func.__name__))
            try:
                result = func(*args, **kwargs)
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.info('{} done. Time: {} s'.format(func.__name__, elapsed_time))
                return result
            except Exception as e:
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.warning('{} error. Time: {} s'.format(func.__name__, elapsed_time), exc_info=True)
            self.logger.debug('{} end'.format(func.__name__))

        return wrapper

    def critical(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            self.logger.debug('{} start'.format(func.__name__))
            try:
                result = func(*args, **kwargs)
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.info('{} done. Time: {} s'.format(func.__name__, elapsed_time))
                # self.cm.send_messange("model run")
                # сделай запись в другую таблицу
                return result
            except Exception as e:
                t_end = time.time()
                elapsed_time = t_end - t_start
                self.logger.critical('{} error. Time: {} s'.format(func.__name__, elapsed_time), exc_info=True)

                log = '{} error. Time: {} s'.format(func.__name__, elapsed_time)
                tb = traceback.format_exc()

                hdfs_logfile = f"""\
                {self.cr.info['model_store_root']}\
                /{self.cr.info['log_folder']}\
                /{os.environ['MODEL_NAME']}\
                /{os.environ['MODEL_VERSION'].replace('.', '_')}\
                /{os.environ['EXECUTION_DATE']}"""

                self.cm.send_messange(f"""Log: \n {log} \n Traceback: \n {tb}""")
            finally:
                self.logger.debug('{} end'.format(func.__name__))
                self.hs.upload_file(self.config['local_log_file'], hdfs_logfile)

        return wrapper
