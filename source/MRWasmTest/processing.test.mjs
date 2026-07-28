import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const before = meshToGeometry( m, false ).indices.length / 3;

  const s = new ml.RemeshSettings();
  s.targetEdgeLen = 0.5;
  assert.equal( ml.remesh( m, s ), true, 'remesh completes' );
  const after = meshToGeometry( m, false ).indices.length / 3;
  assert.ok( after > before, 'remesh subdivides long edges' );

  s.delete();
  m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const p = new ml.MeshRelaxParams();
  p.iterations = 2;
  assert.equal( ml.relax( m, p ), true, 'relax completes' );
  assert.equal( ml.relaxKeepVolume( m, p ), true, 'relaxKeepVolume completes' );
  assert.ok( meshToGeometry( m, false ).positions.length > 0, 'mesh survives relaxation' );

  p.delete();
  m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const p = new ml.FixMeshDegeneraciesParams();
  p.mode = ml.FixMeshDegeneraciesMode.Decimate;
  ml.fixMeshDegeneracies( m, p );
  ml.fixMultipleEdges( m );
  assert.ok( meshToGeometry( m, false ).positions.length > 0, 'mesh survives fixers' );

  p.delete();
  m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const before = meshToGeometry( m, false ).indices.length / 3;

  const s = new ml.SubdivideSettings();
  s.maxEdgeLen = 0.5;
  s.maxEdgeSplits = 10000;
  const splits = ml.subdivideMesh( m, s );
  assert.ok( splits > 0, 'subdivideMesh performs edge splits' );
  assert.ok( meshToGeometry( m, false ).indices.length / 3 > before, 'subdivideMesh increases the face count' );

  s.delete();
  m.delete();
}
