import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.makeTorus(1.0, 0.1, 16, 16);

// Relax mesh (5 iterations)
using relaxParams = new ml.MeshRelaxParams();
relaxParams.iterations = 5;
ml.relax(mesh, relaxParams);

// Subdivide mesh
using subdivideSettings = new ml.SubdivideSettings();
subdivideSettings.maxDeviationAfterFlip = 0.5;
ml.subdivideMesh(mesh, subdivideSettings);

// Rotate mesh
using rotation = ml.Matrix3f.rotation({ x: 0, y: 0, z: 1 }, Math.PI * 0.5);
using rotationXf = ml.AffineXf3f.linear(rotation);
mesh.transform(rotationXf);
