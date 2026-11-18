import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const vn = ml.computePerVertNormals( m );
  assert.equal( vn.toArray().length, 8 * 3, 'one normal per vertex' );
  vn.delete();

  const fn = ml.computePerFaceNormals( m );
  const fnArr = fn.toArray();
  assert.equal( fnArr.length, 12 * 3, 'one normal per face' );
  assert.ok( Math.abs( Math.hypot( fnArr[0], fnArr[1], fnArr[2] ) - 1 ) < 1e-4, 'face normal is unit length' );
  fn.delete();

  m.delete();
}
