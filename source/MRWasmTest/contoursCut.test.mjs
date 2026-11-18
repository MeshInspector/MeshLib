import assert from 'node:assert/strict';
import { ml } from './helpers.mjs';

// precise-intersection pipeline: getVectorConverters -> findCollidingEdgeTrisPrecise ->
// orderIntersectionContours -> getOneMeshIntersectionContours -> cutMesh
{
  const a = ml.makeUVSphere( 1, 32, 32 );
  const b = ml.makeUVSphere( 1, 32, 32 );
  const t = ml.AffineXf3f.translation( { x: 1, y: 0, z: 0 } );
  b.transform( t );  // overlap a along x so the two spheres intersect in a circle
  t.delete();

  const conv = ml.getVectorConverters( a, b );
  const collisions = ml.findCollidingEdgeTrisPrecise( a, b, conv );

  const topoA = a.topology, topoB = b.topology;
  const contours = ml.orderIntersectionContours( topoA, topoB, collisions );
  const omc = ml.getOneMeshIntersectionContours( a, b, contours, true, conv );

  const params = new ml.CutMeshParameters();
  const res = ml.cutMesh( a, omc, params );

  assert.ok( res.resultCut.length > 0, 'cut produced at least one edge path' );
  let cutEdges = 0;
  for ( const path of res.resultCut )
    cutEdges += path.length;
  assert.ok( cutEdges > 0, 'cut edge paths contain edges' );
  assert.ok( Number.isInteger( res.fbsWithContourIntersections.count() ), 'fbsWithContourIntersections is a FaceBitSet' );

  res.fbsWithContourIntersections.delete();
  params.delete();
  omc.delete();
  contours.delete();
  topoA.delete();
  topoB.delete();
  collisions.delete();
  conv.delete();
  a.delete();
  b.delete();
}

// cutMeshByProjection: project a small closed square down onto a sphere cap and cut it
{
  const sphere = ml.makeUVSphere( 1, 48, 48 );
  const square = new Float32Array( [
    0.25, 0.25, 2,
    -0.25, 0.25, 2,
    -0.25, -0.25, 2,
    0.25, -0.25, 2,
    0.25, 0.25, 2,  // closed
  ] );

  const settings = new ml.CutByProjectionSettings();
  settings.direction = { x: 0, y: 0, z: -1 };

  const paths = ml.cutMeshByProjection( sphere, [ square ], settings );
  assert.ok( paths.length > 0, 'projection cut produced at least one edge path' );
  let edges = 0;
  for ( const p of paths )
    edges += p.length;
  assert.ok( edges > 0, 'projected cut path contains edges' );

  settings.delete();
  sphere.delete();
}
