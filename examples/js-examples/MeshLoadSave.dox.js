import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh (a load error throws UnexpectedResultException with the message)
let mesh;
try {
  mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');
} catch (e) {
  console.log(`Failed to load mesh: ${e.message}`);
  process.exit(1);
}
console.log(`Loaded ${mesh.topology.numValidVerts()} vertices, ${mesh.topology.numValidFaces()} faces`);

// Save mesh (the format is chosen from the file extension)
try {
  ml.MeshSave.toAnySupportedFormat(mesh, 'mesh.ply');
} catch (e) {
  console.log(`Failed to save mesh: ${e.message}`);
  process.exit(1);
}
console.log('Saved mesh.ply');
mesh.delete();
