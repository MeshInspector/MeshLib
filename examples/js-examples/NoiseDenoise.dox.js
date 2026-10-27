import * as ml from '@meshinspector/meshlib';

await ml.init();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// add noise to the mesh, scaled to its bounding box
using noiseSettings = new ml.NoiseSettings();
noiseSettings.sigma = mesh.computeBoundingBox().diagonal() * 0.0001;
ml.addNoise(mesh, null, noiseSettings);
mesh.invalidateCaches();
ml.MeshSave.toAnySupportedFormat(mesh, 'noised_mesh.stl');

// denoise the mesh, keeping sharp edges sharp
// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
ml.meshDenoiseViaNormals(mesh);
ml.MeshSave.toAnySupportedFormat(mesh, 'denoised_mesh.stl');
