import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

// meshToPointCloud + PointCloud accessors / addPoint / bbox
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const pc = ml.meshToPointCloud( m, true );

  const pts = pc.points;
  assert.equal( pts.toArray().length, 8 * 3, 'cube becomes 8 points' );
  pts.delete();

  const valid = pc.validPoints;
  assert.equal( valid.count(), 8, '8 valid points' );
  valid.delete();

  const id = pc.addPoint( { x: 0, y: 0, z: 0 } );
  assert.equal( id, 8, 'addPoint returns the new vertex id' );
  const valid2 = pc.validPoints;
  assert.equal( valid2.count(), 9, 'after addPoint there are 9 valid points' );
  valid2.delete();

  const bb = pc.computeBoundingBox();
  assert.ok( bb.valid(), 'point-cloud bounding box is valid' );
  bb.delete();

  pc.delete();
  m.delete();
}

// triangulatePointCloud reconstructs a mesh from a sphere point cloud
{
  const sphere = ml.makeUVSphere( 1, 16, 16 );
  const pc = ml.meshToPointCloud( sphere, true );

  const params = new ml.TriangulationParameters();
  const recon = ml.triangulatePointCloud( pc, params );
  assert.notEqual( recon, null, 'triangulation produced a mesh' );
  const g = meshToGeometry( recon, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'reconstructed mesh is non-empty' );

  recon.delete();
  params.delete();
  pc.delete();
  sphere.delete();
}

// pointUniformSampling + writable validPoints + invalidateCaches
{
  const sphere = ml.makeUVSphere( 1, 32, 32 );
  const pc = ml.meshToPointCloud( sphere, true );

  const initial = pc.validPoints;
  const total = initial.count();
  initial.delete();

  const settings = new ml.UniformSamplingSettings();
  settings.distance = 0.2;
  const samples = ml.pointUniformSampling( pc, settings );
  assert.ok( samples.count() > 0 && samples.count() < total, 'uniform sampling selects a proper subset' );

  pc.validPoints = samples;
  pc.invalidateCaches();
  const after = pc.validPoints;
  assert.equal( after.count(), samples.count(), 'the sampled subset becomes the valid points' );

  after.delete();
  samples.delete();
  settings.delete();
  pc.delete();
  sphere.delete();
}
