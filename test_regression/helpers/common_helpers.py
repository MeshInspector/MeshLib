def list_compare_2d(list1, list2):
    """
    Compare two lists of lists
    Return True if lists are equal, False otherwise.
    :param list1: First list to compare
    :param list2: Second list to compare
    :return: True if lists are equal, False otherwise.
    :rtype: bool
    """
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        assert len(list1[i]) == len(list2[i])
        for j in range(len(list1[i])):
            if list1[i][j] != list2[i][j]:
                return False
    return True
