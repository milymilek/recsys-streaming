import logging


def log_wrapper(enter: str = "Entering", exit: str = "Exiting", log_fn=print):
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_fn("[%s]: %s", func.__name__, enter)
            retval = func(*args, **kwargs)
            log_fn("[%s]: %s", func.__name__, exit)
            return retval

        return wrapper

    return decorator
