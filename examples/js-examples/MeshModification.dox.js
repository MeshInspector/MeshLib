import * as ml from '@meshinspector/meshlib';

await ml.init();

using mesh = ml.makeTorus();

// Relax mesh (5 iterations)
using relaxParams = new ml.MeshRelaxParams();
relaxParams.iterations = 5;
ml.relax(mesh, relaxParams);

// Subdivide mesh
using subdivideSettings = new ml.SubdivideSettings();
subdivideSettings.maxDeviationAfterFlip = 0.5;
ml.subdivideMesh(mesh, subdivideSettings);

// Rotate mesh
const rotationXf = ml.AffineXf3f.linear(ml.Matrix3f.rotation(ml.Vector3f.plusZ(), Math.PI * 0.5));
mesh.transform(rotationXf);
