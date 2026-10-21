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
newMesh.points = mrmesh.VertCoords([mrmesh.Vector3f(0, 0, 0), mrmesh.Vector3f(1, 0, 0), mrmesh.Vector3f(0, 1, 0)])
newMesh.topology = mrmesh.topologyFromTriangles([mrmesh.ThreeVertIds( mrmesh.VertId(0),mrmesh.VertId(1),mrmesh.VertId(2) )])

print(newMesh.topology.getValidFaces().count())