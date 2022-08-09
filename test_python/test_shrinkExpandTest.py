from helper import *
import pytest

def test_shrinkExpand():
	torus = mrmesh.make_components_test_torus(2, 1, 10, 10, None)

	components = mrmesh.get_mesh_components_verts(torus, None)

	assert(len(components) == 5)

	prevCount = components[0].count()
	assert(prevCount != 0)

	for v in range(components[0].size()):
		if components[0].test(mrmesh.VertId(v)):
			components[0].set(mrmesh.VertId(v),False)
			break;

	assert(components[0].count() == prevCount -1)
	prevCount = components[0].count()
	mrmesh.shrink_verts(torus.topology,components[0],1)
	assert(components[0].count() < prevCount)

	prevCount = components[0].count()
	mrmesh.expand_verts(torus.topology,components[0],1)
	assert(components[0].count() > prevCount)
