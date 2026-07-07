import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const eq = ( a, b ) => Math.abs( a - b ) < 1e-4;

  const r = ml.findProjection( { x: 5, y: 0, z: 0 }, m );

  assert.equal( r.valid, true, 'a projection was found' );
  assert.ok( eq( r.distSq, 16 ), 'squared distance from (5,0,0) to the +x face is 16' );
  assert.ok( eq( r.proj.point.x, 1 ) && eq( r.proj.point.y, 0 ) && eq( r.proj.point.z, 0 ),
    'closest surface point is (1,0,0)' );
  assert.ok( Number.isInteger( r.proj.face ) && r.proj.face >= 0, 'proj.face is a valid face index' );
  assert.ok( Number.isInteger( r.mtp.e ), 'mtp.e is an edge index' );
  assert.ok( typeof r.mtp.bary.a === 'number' && typeof r.mtp.bary.b === 'number',
    'barycentric coordinates are present' );

  m.delete();
}
