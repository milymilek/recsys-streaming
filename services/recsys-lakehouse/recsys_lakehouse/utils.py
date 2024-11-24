import logging


def log_wrapper(enter: str = "Entering", exit: str = "Exiting"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info("[%s]: %s", func.__name__, enter)
            retval = func(*args, **kwargs)
            logging.info("[%s]: %s", func.__name__, exit)
            return retval

        return wrapper

    return decorator
