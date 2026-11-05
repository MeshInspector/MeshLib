import * as ml from '@meshinspector/meshlib';

await ml.init();

// create first sphere with radius of 1 unit
using sphere1 = ml.makeUVSphere(1.0, 64, 64);

// create second sphere by cloning the first sphere and moving it in X direction
using sphere2 = new ml.Mesh(sphere1);
sphere2.transform(ml.AffineXf3f.translation(new ml.Vector3f(0.7, 0.0, 0.0)));

// optional mapper relating the primitives of the input meshes to the primitives of the result
using mapper = new ml.BooleanResultMapper();
using params = new ml.BooleanParameters();
params.mapper = mapper;

// perform boolean operation
using result = ml.boolean(sphere1, sphere2, ml.BooleanOperation.Intersection, params);
if (!result.valid())
  throw new Error(result.errorString.toString());

// find the faces of the result produced by each input sphere, and the faces the cut created
const MapObject = ml.BooleanResultMapper_MapObject;
using facesOfSphere1 = mapper.map(sphere1.topology.getValidFaces(), MapObject.A);
using facesOfSphere2 = mapper.map(sphere2.topology.getValidFaces(), MapObject.B);
using newFaces = mapper.newFaces();
console.log(`faces from sphere1: ${facesOfSphere1.count()}`);
console.log(`faces from sphere2: ${facesOfSphere2.count()}`);
console.log(`faces created by the cut: ${newFaces.count()}`);

// map one particular face of sphere1 forward: the cut can split it in several faces of the
// result, or drop it completely if that part of sphere1 is not in the result
const faceOfSphere1 = new ml.FaceId(793);
using oneFace = new ml.FaceBitSet();
oneFace.autoResizeSet(faceOfSphere1, true);
using producedFaces = mapper.map(oneFace, MapObject.A);
console.log(`face ${faceOfSphere1.id_} of sphere1 produced ${producedFaces.count()} faces of the result`);

// and backward: the face of sphere1 each face of the result came from
// (an invalid id for the faces that came from sphere2)
using new2OldFaces = mapper.getNew2OldFaceMap(MapObject.A);
const resultFace = producedFaces.find_first();
console.log(`face ${resultFace.id_} of the result came from face ${new2OldFaces.index(resultFace).id_} of sphere1`);

// save result to STL file
ml.MeshSave.toAnySupportedFormat(result.mesh, 'out_boolean.stl');
