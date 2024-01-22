import numpy as np
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy


def createTestMesh() -> mrmeshpy.Mesh:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).astype(np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)

    # Basic assertions to verify the mesh is created correctly
    assert mesh.topology.numValidFaces() == 2
    assert mesh.topology.numValidVerts() == 4

    return mesh

def test_basicHalfEdgeOperations():
    mesh = createTestMesh()

    # Edge 0: 0 -> 1
    edge = mrmeshpy.EdgeId(0)

    assert mesh.topology.hasEdge(edge)
    assert not mesh.topology.hasEdge(mrmeshpy.EdgeId(100))

    assert mesh.topology.dest(edge).get() == 1 # Goes to vertex 1
    assert mesh.topology.org(edge).get() == 0 # Comes from vertex 0

    assert mesh.topology.prev(edge).get() == 9 # Previous edge is 9
    assert mesh.topology.next(edge).get() == 5 # Next edge is 1

    assert mesh.topology.left(edge).get() == 0 # Left face is 0
    assert mesh.topology.right(edge).get() == -1 # Has no right face

    assert mesh.topology.getOrgDegree(edge) == 3
    assert mesh.topology.getVertDegree(mrmeshpy.VertId(0)) == 3
    assert mesh.topology.getLeftDegree(edge) == 3
    assert mesh.topology.getFaceDegree(mrmeshpy.FaceId(0)) == 3