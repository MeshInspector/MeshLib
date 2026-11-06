import pytest
from helper import *

def test_unite_close_vertices():
	positions = [
	mrmesh.Vector3f(0,0,0),
	mrmesh.Vector3f(0,1,0),
	mrmesh.Vector3f(1,1,0),
	mrmesh.Vector3f(1,0,0),
	mrmesh.Vector3f(0,0,1),
	mrmesh.Vector3f(0,1,1),
	mrmesh.Vector3f(1,1,1),
	mrmesh.Vector3f(1,0,1)
	]
	# 12 separate triangles, each with its own copies of the cube corners
	corners = [0,1,2, 2,3,0, 0,4,5, 5,1,0, 0,3,7, 7,4,0, 6,5,4, 4,7,6, 1,5,6, 6,2,1, 6,7,3, 3,2,6]
	mesh = mrmesh.Mesh()
	mesh.points = mrmesh.VertCoords([positions[c] for c in corners])
	mesh.topology = mrmesh.topologyFromTriangles(
		[mrmesh.ThreeVertIds([mrmesh.VertId(i*3 + 0),mrmesh.VertId(i*3 + 1),mrmesh.VertId(i*3 + 2)]) for i in range(12)])

	assert mrmesh.getAllComponents(mesh).size() == 12
	assert mesh.topology.findHoleRepresentiveEdges().size() == 12

	mrmesh.uniteCloseVertices(mesh,0)
	assert mrmesh.getAllComponents(mesh).size() == 1
	assert mesh.topology.findHoleRepresentiveEdges().size() == 0
