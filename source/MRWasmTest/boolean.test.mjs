import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

function isClosedGenus0( verts, tris ) {
  return tris % 2 === 0 && 2 * verts - tris === 4;
}

{
  const A = cube( 0, 0, 0, 2 );
  const B = cube( 1, 1, 1, 2 );
  const ma = meshFromGeometry( A.positions, A.indices );
  const mb = meshFromGeometry( B.positions, B.indices );

  const res = ml.boolean( ma, mb, ml.BooleanOperation.Union );
  assert.ok( res.valid(), 'union succeeded' );
  const u = res.mesh;

  const gBefore = meshToGeometry( u, false );
  const vBefore = gBefore.positions.length / 3;
  const fBefore = gBefore.indices.length / 3;
  assert.ok( fBefore > 12, 'union has more triangles than a single cube' );
  assert.ok( isClosedGenus0( vBefore, fBefore ),
    `union should be a closed genus-0 solid (V=${vBefore}, F=${fBefore})` );

  const s = new ml.DecimateSettings();
  s.maxDeletedFaces = Math.floor( fBefore / 2 );
  const dr = ml.decimateMesh( u, s );
  assert.ok( dr.facesDeleted >= 0, 'facesDeleted is non-negative' );
  assert.ok( dr.facesDeleted <= Math.floor( fBefore / 2 ), 'decimate honors maxDeletedFaces' );

  const gAfter = meshToGeometry( u, false );
  const fAfter = gAfter.indices.length / 3;
  const vAfter = gAfter.positions.length / 3;
  assert.ok( fAfter <= fBefore, 'decimate never increases triangle count' );
  assert.ok( fAfter >= 4, 'decimate does not destroy the mesh' );
  for ( let i = 0; i < gAfter.indices.length; i++ )
    assert.ok( gAfter.indices[i] < vAfter, 'all indices in range after decimate' );

  const gn = meshToGeometry( u, true );
  assert.ok( gn.normals instanceof Float32Array && gn.normals.length === gn.positions.length,
    'normals array parallels positions' );

  s.delete(); u.delete(); res.delete(); ma.delete(); mb.delete();
}

{
  const A = cube( 0, 0, 0, 2 );
  const Far = cube( 100, 100, 100, 2 );
  const ma = meshFromGeometry( A.positions, A.indices );
  const mf = meshFromGeometry( Far.positions, Far.indices );

  const res = ml.boolean( ma, mf, ml.BooleanOperation.Intersection );
  assert.ok( res.valid(), 'disjoint intersection is still valid' );
  const x = res.mesh;
  assert.equal( meshToGeometry( x, false ).indices.length, 0, 'intersection of disjoint meshes is empty' );

  x.delete(); res.delete(); ma.delete(); mf.delete();
}
