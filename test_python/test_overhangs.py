import pytest
from helper import *


def test_overhangs():
    torusbase = mrmesh.makeTorus(2, 1, 32, 32, None)
    torustop = mrmesh.makeTorus(2, 1, 32, 32, None)
    torustop.transform(mrmesh.AffineXf3f.translation(mrmesh.Vector3f(0, 0, 3.0)))

    mergedMesh = mrmesh.mergeMeshes([torusbase, torustop])

    oParams = mrmesh.FindOverhangsSettings()
    oParams.layerHeight = 0.1
    oParams.maxOverhangDistance = 0.1
    overhangs = mrmesh.findOverhangs(mergedMesh, oParams)
    # if base has Z size bigger than one layer than it is overhang too
    # however the base layer is not considered as an overhang,
    # thus the overhang is split into outer and inner parts
    assert (
        overhangs.size() == 3
    )
