import os
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


def compare_files(file1: str or Path, file2: str or Path):
    """
    Compare two files and return True if they are equal
    :param file1: first file to compare
    :param file2: second file to compare
    :return: True if the files are equal, False otherwise
    """
    if not os.path.exists(file1):
        logger.error(f"File {file1} does not exist")
        return False
    if not os.path.isfile(file2):
        logger.error(f"File {file2} does not exist")
        return False

    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    if size1 != size2:
        return False

    # algorithm below valid only if files have same size (it's checked above)
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        byte1 = f1.read(1)
        byte2 = f2.read(1)
        while byte1 and byte2:
            if byte1 != byte2:
                return False
            byte1 = f1.read(1)
            byte2 = f2.read(1)
    return True
