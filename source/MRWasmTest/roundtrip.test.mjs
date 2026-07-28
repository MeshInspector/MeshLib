import assert from 'node:assert/strict';
import { cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions instanceof Float32Array, 'positions is a Float32Array' );
  assert.ok( g.indices instanceof Uint32Array, 'indices is a Uint32Array' );
  assert.equal( g.positions.length, 8 * 3, 'cube welds to 8 vertices' );
  assert.equal( g.indices.length, 12 * 3, 'cube has 12 triangles' );

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
