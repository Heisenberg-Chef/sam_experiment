import logging

# initialize logger
logger_train = logging.getLogger("TRAINING")
logger_train.setLevel(logging.DEBUG)

logger_val = logging.getLogger("EVALUATE")
logger_val.setLevel(logging.DEBUG)

# rotating file handler setting.
file_handler = logging.FileHandler("train.log")
console_handler = logging.StreamHandler()

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s")

file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)

logger_train.addHandler(file_handler)
logger_val.addHandler(console_handler)

logger_train = logger_train
logger_val = logger_val
