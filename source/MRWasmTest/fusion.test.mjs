import assert from 'node:assert/strict';
import { ml, meshToGeometry } from './helpers.mjs';

// build a point cloud from a sphere, then reconstruct a mesh via fusion
{
  const sphere = ml.makeUVSphere( 1, 32, 32 );
  const pc = ml.meshToPointCloud( sphere, true );

  const radius = ml.findAvgPointsRadius( pc, 50 );
  assert.ok( radius > 0, 'findAvgPointsRadius returns a positive radius' );

  const bbox = pc.computeBoundingBox();
  const params = new ml.PointsToMeshParameters();
  params.voxelSize = bbox.diagonal() * 0.02;
  params.sigma = Math.max( params.voxelSize, radius );
  params.minWeight = 1;
  bbox.delete();

  const mesh = ml.pointsToMeshFusion( pc, params );
  const g = meshToGeometry( mesh, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'fusion reconstructs a non-empty mesh' );

  mesh.delete();
  params.delete();
  pc.delete();
  sphere.delete();
}
