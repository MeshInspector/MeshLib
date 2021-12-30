print('Import mrmesh')
import mrmeshpy as mrmesh

print('Start test script')

mesh = mrmesh.load_mesh("M:\\sectioning\\sectioning\\468\\Final Output\\P523468_101_OtherFiles\\stitched.mrmesh")
holes = mesh.topology.findHoleRepresentiveEdges()
vec3 = mrmesh.Vector3();
vec3.z = 1.0;
fbs = mrmesh.FaceBitSet()
newEdge = mrmesh.build_bottom(mesh,holes[0],vec3,0.1,fbs)

print(fbs.count())

params = mrmesh.FillHoleParams()
faces = mrmesh.FaceBitSet()
params.outNewFaces = faces
print(params.outNewFaces.count())

mrmesh.fill_hole(mesh,newEdge,params)

print(params.outNewFaces.count())

direct = mrmesh.Vector3()
direct.x = 1
direct.y = 0
direct.z = 0
direct = direct.normalized()

up = mrmesh.Vector3()
up.x = 0
up.y = 1
up.z = 0

newMesh = mrmesh.Mesh();
newMesh.points.vec.resize(3)
newMesh.points.vec[1].x = 1
newMesh.points.vec[2].y = 1

tris = mrmesh.vecMeshBuilderTri()
tris.resize(1)
tris[0] = mrmesh.MeshBuilderTri(mrmesh.VertId(0),mrmesh.VertId(1),mrmesh.VertId(2),mrmesh.FaceId(0))
newMesh.topology = mrmesh.topologyFromTriangles(tris)
