import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// add noise to the mesh, scaled to its bounding box
using bbox = mesh.computeBoundingBox();
using noiseSettings = new ml.NoiseSettings();
noiseSettings.sigma = bbox.diagonal() * 0.0001;
ml.addNoise(mesh, noiseSettings);
mesh.invalidateCaches();
ml.MeshSave.toAnySupportedFormat(mesh, 'noised_mesh.stl');

// denoise the mesh, keeping sharp edges sharp
// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
ml.meshDenoiseViaNormals(mesh);
ml.MeshSave.toAnySupportedFormat(mesh, 'denoised_mesh.stl');
