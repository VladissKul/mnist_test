class LoggerMessanges:

    def __init__(self):
        pass

    def log_debug(self, msg: str = '') -> None:
        self.logger.debug(msg)

    def log_info(self, msg: str = '') -> None:
        self.logger.info(msg)

    def log_warning(self, msg: str = '') -> None:
        self.logger.warning(msg)

    def log_error(self, msg: str = '') -> None:
        self.logger.error(msg)

    def log_critical(self, msg: str = '') -> None:
        self.logger.critical(msg)
