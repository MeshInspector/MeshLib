from helper import *
import pytest


def test_overhangs():
	torusbase = mrmesh.makeTorus(2, 1, 32, 32, None)
	torustop = mrmesh.makeTorus(2, 1, 32, 32, None)
	torustop.transform( mrmesh.AffineXf3f.translation(mrmesh.Vector3f(0,0,3.0)) )

	mergedMesh = mrmesh.mergeMehses([torusbase, torustop ])

	oParams = mrmesh.FindOverhangsSettings()
	oParams.layerHeight = 0.1
	oParams.maxOverhangDistance = 0.1
	overhangs = mrmesh.findOverhangs(mergedMesh,oParams)
	assert (overhangs.size() == 1)