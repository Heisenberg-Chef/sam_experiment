import logging
from logging.handlers import TimedRotatingFileHandler


class logger:
    """
    custom logger
    """

    def __init__(self, log_name: str = "test", log_level: int = logging.INFO):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        # formatter
        formatter = logging.Formatter(r'%(asctime)s %(levelname)-5s --- [%(threadName)+15s] %(name)-40s : %(message)s')
        # console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        # file handler
        file_handler = logging.FileHandler(log_name)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == '__main__':
    log = logger()
    log.error("hello world")
