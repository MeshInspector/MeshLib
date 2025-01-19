"""
Set of tests to check if the bindings not crash on wrong arguments types.
No corresponding issue was formally reported, but such crashes happened at some moment.
"""
import meshlib.mrmeshpy as mr
import pytest
from module_helper import *


@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_issue_type_error():
    with pytest.raises(TypeError):
        mr.findTwinEdgePairs("1", 0)

@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_issue_type_error_wrong_args_num():
    with pytest.raises(TypeError):
        mr.findTwinEdgePairs("1")

@pytest.mark.smoke
def test_issue_type_error_custom_binding():
    with pytest.raises(TypeError):
        mr.loadMesh(1)
