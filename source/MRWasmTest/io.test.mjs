import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rmSync } from 'node:fs';
import { ml } from './helpers.mjs';

{
  const sphere = ml.makeUVSphere( 1, 16, 16 );

  // mesh round-trip
  const topo0 = sphere.topology;
  const v0 = topo0.getValidVerts(), f0 = topo0.getValidFaces();
  const nv = v0.count(), nf = f0.count();
  v0.delete(); f0.delete(); topo0.delete();

  const meshPath = join( tmpdir(), `meshlib-io-${process.pid}.ply` );
  ml.MeshSave.toAnySupportedFormat( sphere, meshPath );
  const loaded = ml.MeshLoad.fromAnySupportedFormat( meshPath );
  const topo1 = loaded.topology;
  const v1 = topo1.getValidVerts(), f1 = topo1.getValidFaces();
  assert.equal( v1.count(), nv, 'mesh round-trip preserves vertex count' );
  assert.equal( f1.count(), nf, 'mesh round-trip preserves face count' );
  v1.delete(); f1.delete(); topo1.delete();
  loaded.delete();
  rmSync( meshPath );

  // point-cloud round-trip
  const pc = ml.meshToPointCloud( sphere, true );
  const valid0 = pc.validPoints;
  const np = valid0.count();
  valid0.delete();

  const pcPath = join( tmpdir(), `meshlib-io-${process.pid}-pts.ply` );
  ml.PointsSave.toAnySupportedFormat( pc, pcPath );
  const loadedPc = ml.PointsLoad.fromAnySupportedFormat( pcPath );
  const valid1 = loadedPc.validPoints;
  assert.equal( valid1.count(), np, 'point-cloud round-trip preserves point count' );
  valid1.delete();
  loadedPc.delete();
  pc.delete();
  rmSync( pcPath );

  sphere.delete();
}
