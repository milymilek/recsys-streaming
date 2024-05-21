from .download_data import run as download_data
from .extract_presentation_data import run as extract_presentation_data
from .insert_in_db import run as insert_in_db
from .process_data import run as process_data

__all__ = [download_data, extract_presentation_data, insert_in_db, process_data]