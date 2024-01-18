import os
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)

MULTIPLE_REFERENCE_FILES_SUFFIX = "_ref"


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


def compare_file_with_multiple_references(test_file, refs_list):
    """
    Compare input file with multiple references
    :param test_file: file to compare
    :param refs_list: list of references to compare
    :return: True if the input file is equal to one of reference files
    """
    for ref in refs_list:
        if compare_files(test_file, ref):
            return True
    return False


def get_reference_files_list(full_file_path: Path):
    """
    Get list of reference files for the test file. For file foo.bar it searches for foo_opt1.bar, foo_opt2.bar, etc.
    :param full_file_path: name of the test file
    :return: list of reference files
    """
    references_list = [full_file_path]
    for i in full_file_path.parent.glob(f"{full_file_path.stem}{MULTIPLE_REFERENCE_FILES_SUFFIX}*"):
        logger.debug(f"Found reference file {i}")
        references_list.append(i)
    return references_list
