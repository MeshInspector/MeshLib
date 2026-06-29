import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

// an open box (cube minus its +z face) has one hole that fillHole closes
{
  const c = cube( 0, 0, 0, 2 );
  const idx = Array.from( c.indices );
  // the +z face is triangles 2 and 3 (flat indices 6..11); drop them
  const holed = new Uint32Array( [ ...idx.slice( 0, 6 ), ...idx.slice( 12 ) ] );
  const m = meshFromGeometry( c.positions, holed );

  let topo = m.topology;
  assert.equal( topo.findNumHoles(), 1, 'the open box has one hole' );
  const holeEdges = topo.findHoleRepresentiveEdges();
  assert.equal( holeEdges.length, 1, 'one hole representative edge' );
  topo.delete();

  const params = new ml.FillHoleParams();
  params.multipleEdgesResolveMode = ml.MultipleEdgesResolveMode.Simple;
  ml.fillHole( m, holeEdges[ 0 ], params );
  params.delete();

  topo = m.topology;
  assert.equal( topo.findNumHoles(), 0, 'the hole is filled' );
  topo.delete();
  m.delete();
}

// metric factory + calcCombinedFillMetric
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const metric = ml.getCircumscribedMetric( m );
  assert.ok( metric instanceof ml.FillHoleMetric, 'factory returns a FillHoleMetric' );

  const topo = m.topology;
  const faces = topo.getValidFaces();
  const v = ml.calcCombinedFillMetric( m, faces, metric );
  assert.ok( Number.isFinite( v ), 'combined fill metric is a finite number' );

  faces.delete();
  topo.delete();
  metric.delete();
  m.delete();
}
