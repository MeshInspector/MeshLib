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

// FillHoleParams.metric accepts a FillHoleMetric
{
  const c = cube( 0, 0, 0, 2 );
  const idx = Array.from( c.indices );
  const holed = new Uint32Array( [ ...idx.slice( 0, 6 ), ...idx.slice( 12 ) ] );
  const m = meshFromGeometry( c.positions, holed );

  const topo = m.topology;
  const holeEdges = topo.findHoleRepresentiveEdges();
  topo.delete();

  const params = new ml.FillHoleParams();
  const metric = ml.getUniversalMetric( m );
  params.metric = metric;
  ml.fillHole( m, holeEdges[ 0 ], params );
  metric.delete();
  params.delete();

  const topo2 = m.topology;
  assert.equal( topo2.findNumHoles(), 0, 'the hole is filled using the universal metric' );
  topo2.delete();
  m.delete();
}

// stitchHoles — connect the two open ends of a tube with a cylindrical band
{
  const c = cube( 0, 0, 0, 2 );
  const idx = Array.from( c.indices );
  // keep only the four side faces (drop the -z face 0..5 and the +z face 6..11): an open tube
  const tube = new Uint32Array( idx.slice( 12 ) );
  const m = meshFromGeometry( c.positions, tube );

  let topo = m.topology;
  assert.equal( topo.findNumHoles(), 2, 'the open tube has two holes' );
  const holeEdges = topo.findHoleRepresentiveEdges();
  assert.equal( holeEdges.length, 2, 'two hole representative edges' );
  topo.delete();

  const params = new ml.StitchHolesParams();
  const metric = ml.getUniversalMetric( m );
  params.metric = metric;
  ml.stitchHoles( m, holeEdges[ 0 ], holeEdges[ 1 ], params );
  metric.delete();
  params.delete();

  topo = m.topology;
  assert.equal( topo.findNumHoles(), 0, 'stitching connects the two holes (none remain)' );
  topo.delete();
  m.delete();
}
