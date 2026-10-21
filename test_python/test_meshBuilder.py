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
	mesh = mrmesh.Mesh()
	mesh.points.push_back(positions[0])
	mesh.points.push_back(positions[1])
	mesh.points.push_back(positions[2])

	mesh.points.push_back(positions[2])
	mesh.points.push_back(positions[3])
	mesh.points.push_back(positions[0])

	mesh.points.push_back(positions[0])
	mesh.points.push_back(positions[4])
	mesh.points.push_back(positions[5])

	mesh.points.push_back(positions[5])
	mesh.points.push_back(positions[1])
	mesh.points.push_back(positions[0])

	mesh.points.push_back(positions[0])
	mesh.points.push_back(positions[3])
	mesh.points.push_back(positions[7])

	mesh.points.push_back(positions[7])
	mesh.points.push_back(positions[4])
	mesh.points.push_back(positions[0])

	mesh.points.push_back(positions[6])
	mesh.points.push_back(positions[5])
	mesh.points.push_back(positions[4])

	mesh.points.push_back(positions[4])
	mesh.points.push_back(positions[7])
	mesh.points.push_back(positions[6])

	mesh.points.push_back(positions[1])
	mesh.points.push_back(positions[5])
	mesh.points.push_back(positions[6])

	mesh.points.push_back(positions[6])
	mesh.points.push_back(positions[2])
	mesh.points.push_back(positions[1])

	mesh.points.push_back(positions[6])
	mesh.points.push_back(positions[7])
	mesh.points.push_back(positions[3])

	mesh.points.push_back(positions[3])
	mesh.points.push_back(positions[2])
	mesh.points.push_back(positions[6])

	tris = mrmesh.Triangulation()
	for i in range(12):
		tris.push_back(mrmesh.ThreeVertIds([mrmesh.VertId(i*3 + 0),mrmesh.VertId(i*3 + 1),mrmesh.VertId(i*3 + 2)]))
	mesh.topology = mrmesh.topologyFromTriangles(tris)

	assert mrmesh.getAllComponents(mesh).size() == 12
	assert mesh.topology.findHoleRepresentiveEdges().size() == 12

	mrmesh.uniteCloseVertices(mesh,0)
	assert mrmesh.getAllComponents(mesh).size() == 1
	assert mesh.topology.findHoleRepresentiveEdges().size() == 0
