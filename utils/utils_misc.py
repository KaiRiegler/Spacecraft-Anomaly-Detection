import logging
import tensorflow as tf


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):
    '''
    Configures logging for both standard Python logging and TensorFlow logging.

        Parameters:
            path_log (str, optional): Path to save log file. Defaults to None.
            logging_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to 0.
            b_stream (bool, optional): Whether to stream logs to the console. Defaults to False.
            b_debug (bool, optional): Whether to enable TensorFlow debugging log device placement. Defaults to False.
    '''
    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
