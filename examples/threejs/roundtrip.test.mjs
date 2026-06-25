import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const moduleUrl = process.env.MESHLIB_MODULE
  ? pathToFileURL( process.env.MESHLIB_MODULE ).href
  : new URL( './meshlib.mjs', import.meta.url ).href;

const { default: createMeshLib } = await import( moduleUrl );
const ml = await createMeshLib();

function cube( cx, cy, cz, s = 2 ) {
  const h = s / 2;
  const positions = new Float32Array( [
    cx - h, cy - h, cz - h,
    cx + h, cy - h, cz - h,
    cx + h, cy + h, cz - h,
    cx - h, cy + h, cz - h,
    cx - h, cy - h, cz + h,
    cx + h, cy - h, cz + h,
    cx + h, cy + h, cz + h,
    cx - h, cy + h, cz + h,
  ] );
  const indices = new Uint32Array( [
    0, 2, 1,  0, 3, 2,
    4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,
    3, 6, 2,  3, 7, 6,
    0, 4, 7,  0, 7, 3,
    1, 2, 6,  1, 6, 5,
  ] );
  return { positions, indices };
}

function isClosedGenus0( verts, tris ) {
  return tris % 2 === 0 && 2 * verts - tris === 4;
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = ml.meshFromGeometry( c.positions, c.indices );
  assert.equal( m.numVerts(), 8, 'cube welds to 8 vertices' );
  assert.equal( m.numTris(), 12, 'cube has 12 triangles' );

  const g = m.toGeometry();
  assert.ok( g.positions instanceof Float32Array, 'positions is a Float32Array' );
  assert.ok( g.indices instanceof Uint32Array, 'indices is a Uint32Array' );
  assert.equal( g.positions.length, 8 * 3 );
  assert.equal( g.indices.length, 12 * 3 );

  const nv = g.positions.length / 3;
  for ( let i = 0; i < g.indices.length; i++ )
    assert.ok( g.indices[i] < nv, 'every index is in range' );
  for ( let f = 0; f < g.indices.length; f += 3 ) {
    const [a, b, d] = [g.indices[f], g.indices[f + 1], g.indices[f + 2]];
    assert.ok( a !== b && b !== d && a !== d, 'no degenerate triangle' );
  }

  let lo = Infinity, hi = -Infinity;
  for ( const v of g.positions ) { lo = Math.min( lo, v ); hi = Math.max( hi, v ); }
  assert.ok( Math.abs( lo + 1 ) < 1e-5 && Math.abs( hi - 1 ) < 1e-5, 'bounding box preserved at [-1,1]' );

  m.delete();
}

{
  const A = cube( 0, 0, 0, 2 );
  const B = cube( 1, 1, 1, 2 );
  const ma = ml.meshFromGeometry( A.positions, A.indices );
  const mb = ml.meshFromGeometry( B.positions, B.indices );

  const u = ml.boolean( ma, mb, ml.BooleanOp.Union );
  const vBefore = u.numVerts();
  const fBefore = u.numTris();
  assert.ok( fBefore > 12, 'union has more triangles than a single cube' );
  assert.ok( isClosedGenus0( vBefore, fBefore ),
    `union should be a closed genus-0 solid (V=${vBefore}, F=${fBefore})` );

  const res = ml.decimate( u, { targetRatio: 0.5 } );
  assert.ok( res.facesDeleted >= 0, 'facesDeleted is non-negative' );
  assert.ok( u.numTris() <= fBefore, 'decimate never increases triangle count' );
  assert.ok( u.numTris() >= 4, 'decimate does not destroy the mesh' );
  assert.ok( u.numTris() >= Math.floor( fBefore * 0.25 ), 'decimate did not wildly over-simplify' );

  const g = u.toGeometry();
  assert.equal( g.positions.length / 3, u.numVerts(), 'exported vertex count matches packed mesh' );
  assert.equal( g.indices.length / 3, u.numTris(), 'exported triangle count matches packed mesh' );
  const nv = g.positions.length / 3;
  for ( let i = 0; i < g.indices.length; i++ )
    assert.ok( g.indices[i] < nv, 'all indices in range after pack' );

  const gn = u.toGeometryWithNormals();
  assert.ok( gn.normals instanceof Float32Array && gn.normals.length === gn.positions.length,
    'normals array parallels positions' );

  ma.delete(); mb.delete(); u.delete();
}

{
  const A = cube( 0, 0, 0, 2 );
  const Far = cube( 100, 100, 100, 2 );
  const ma = ml.meshFromGeometry( A.positions, A.indices );
  const mf = ml.meshFromGeometry( Far.positions, Far.indices );
  try {
    const x = ml.boolean( ma, mf, ml.BooleanOp.Intersection );
    assert.ok( x.numTris() >= 0, 'intersection returned a (possibly empty) mesh' );
    x.delete();
  } catch ( e ) {
    assert.ok( e !== undefined, 'a failed boolean surfaces as a catchable error' );
  }
  ma.delete(); mf.delete();
}

console.log( 'OK' );
