import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// create first sphere with radius of 1 unit
using sphere1 = ml.makeUVSphere(1.0, 64, 64);

// create second sphere (the JS bindings expose no cheap mesh copy, so regenerate an
// identical sphere) and move it in the X direction
using sphere2 = ml.makeUVSphere(1.0, 64, 64);
using xf = ml.AffineXf3f.translation({ x: 0.7, y: 0.0, z: 0.0 });
sphere2.transform(xf);

// perform boolean operation
using result = ml.boolean(sphere1, sphere2, ml.BooleanOperation.Intersection);
if (!result.valid())
  throw new Error(result.errorString);

// save result to STL file
using resultMesh = result.mesh;
ml.MeshSave.toAnySupportedFormat(resultMesh, 'out_boolean.stl');
