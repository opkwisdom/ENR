import os
import logging

def _setup_logger(output_dir: str):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler('{}/log.txt'.format(output_dir))
    file_handler.setFormatter(log_format)

    logger.handlers = [console_handler, file_handler]

    return logger