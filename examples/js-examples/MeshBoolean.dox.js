import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// create first sphere with radius of 1 unit
using sphere1 = ml.makeUVSphere(1.0, 64, 64);

// create second sphere (the JS bindings expose no cheap mesh copy, so regenerate an
// identical sphere) and move it in the X direction
using sphere2 = ml.makeUVSphere(1.0, 64, 64);
using xf = ml.AffineXf3f.translation({ x: 0.7, y: 0.0, z: 0.0 });
sphere2.transform(xf);

// optional mapper relating the primitives of the input meshes to the primitives of the result
using mapper = new ml.BooleanResultMapper();

// perform boolean operation
using result = ml.boolean(sphere1, sphere2, ml.BooleanOperation.Intersection, mapper);
if (!result.valid())
  throw new Error(result.errorString);

// find the faces of the result produced by each input sphere, and the faces the cut created
using topology1 = sphere1.topology;
using topology2 = sphere2.topology;
using validFaces1 = topology1.getValidFaces();
using validFaces2 = topology2.getValidFaces();
using facesOfSphere1 = mapper.mapFaces(validFaces1, ml.BooleanMapObject.A);
using facesOfSphere2 = mapper.mapFaces(validFaces2, ml.BooleanMapObject.B);
using newFaces = mapper.newFaces();
console.log(`faces from sphere1: ${facesOfSphere1.count()}`);
console.log(`faces from sphere2: ${facesOfSphere2.count()}`);
console.log(`faces created by the cut: ${newFaces.count()}`);

// map one particular face of sphere1 forward: the cut can split it in several faces of the
// result, or drop it completely if that part of sphere1 is not in the result
const faceOfSphere1 = 793;
using oneFace = ml.FaceBitSet.fromIndices([faceOfSphere1]);
using producedFaces = mapper.mapFaces(oneFace, ml.BooleanMapObject.A);
console.log(`face ${faceOfSphere1} of sphere1 produced ${producedFaces.count()} faces of the result`);

// and backward: the face of sphere1 each face of the result came from
// (4294967295, i.e. an invalid id, for the faces that came from sphere2)
using new2OldFaces = mapper.getNew2OldFaceMap(ml.BooleanMapObject.A);
const new2OldFacesArray = new2OldFaces.toArray();
const resultFace = producedFaces.find_first();
console.log(`face ${resultFace} of the result came from face ${new2OldFacesArray[resultFace]} of sphere1`);

// save result to STL file
using resultMesh = result.mesh;
ml.MeshSave.toAnySupportedFormat(resultMesh, 'out_boolean.stl');
