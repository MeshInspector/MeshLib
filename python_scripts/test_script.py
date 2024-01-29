print('Import mrmesh')
import mrmeshpy as mrmesh

print('Start test script')

mesh = mrmesh.loadMesh("M:\\sectioning\\sectioning\\468\\Final Output\\P523468_101_OtherFiles\\stitched.mrmesh")
holes = mesh.topology.findHoleRepresentiveEdges()
vec3 = mrmesh.Vector3f();
vec3.z = 1.0;
fbs = mrmesh.FaceBitSet()
newEdge = mrmesh.buildBottom(mesh,holes[0],vec3,0.1,fbs)

print(fbs.count())

params = mrmesh.FillHoleParams()
faces = mrmesh.FaceBitSet()
params.outNewFaces = faces
print(params.outNewFaces.count())

mrmesh.fillHole(mesh,newEdge,params)

print(params.outNewFaces.count())

direct = mrmesh.Vector3f()
direct.x = 1
direct.y = 0
direct.z = 0
direct = direct.normalized()

up = mrmesh.Vector3f()
up.x = 0
up.y = 1
up.z = 0

newMesh = mrmesh.Mesh();
newMesh.points.vec.resize(3)
newMesh.points.vec[1].x = 1
newMesh.points.vec[2].y = 1

tris = mrmesh.Triangulation()
tris.vec.resize(1)
tris.vec[0] = mrmesh.ThreeVertIds( mrmesh.VertId(0),mrmesh.VertId(1),mrmesh.VertId(2) )
newMesh.topology = mrmesh.topologyFromTriangles(tris)

print(newMesh.topology.getValidFaces().count())