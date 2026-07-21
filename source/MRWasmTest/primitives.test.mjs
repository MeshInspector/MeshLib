import assert from 'node:assert/strict';
import { ml } from './helpers.mjs';

{
  const I = ml.Matrix3f.identity();
  assert.ok( I.x.x === 1 && I.x.y === 0 && I.x.z === 0, 'identity row x is (1,0,0)' );
  assert.ok( I.y.y === 1 && I.z.z === 1, 'identity has unit diagonal' );
  I.delete();

  const R = ml.Matrix3f.rotation( { x: 0, y: 0, z: 1 }, Math.PI / 2 );
  assert.ok( Number.isFinite( R.x.x ) && Number.isFinite( R.y.x ), 'rotation matrix entries are finite' );
  R.delete();

  const t = ml.AffineXf3f.translation( { x: 1, y: 2, z: 3 } );
  const p = t.apply( { x: 0, y: 0, z: 0 } );
  assert.ok( p.x === 1 && p.y === 2 && p.z === 3, 'translation maps origin to (1,2,3)' );
  const t2 = ml.AffineXf3f.translation( { x: 10, y: 0, z: 0 } );
  const composed = t.mul( t2 );
  const p2 = composed.apply( { x: 0, y: 0, z: 0 } );
  assert.ok( p2.x === 11 && p2.y === 2 && p2.z === 3, 'composed translations add up' );
  t.delete(); t2.delete(); composed.delete();

  const box = new ml.Box3i( { x: 0, y: 0, z: 0 }, { x: 2, y: 3, z: 4 } );
  assert.equal( box.volume(), 24, 'integer box volume is 2*3*4' );
  const bsz = box.size();
  assert.ok( bsz.x === 2 && bsz.y === 3 && bsz.z === 4, 'integer box size is (2,3,4)' );
  box.delete();
}

// Vector3 helpers operate on plain {x,y,z} value objects
{
  const z = ml.Vector3.plusZ();
  assert.ok( z.x === 0 && z.y === 0 && z.z === 1, 'plusZ is (0,0,1)' );

  const d = ml.Vector3.diagonal( 2 );
  assert.ok( d.x === 2 && d.y === 2 && d.z === 2, 'diagonal(2) is (2,2,2)' );

  const s = ml.Vector3.add( { x: 1, y: 2, z: 3 }, { x: 10, y: 20, z: 30 } );
  assert.ok( s.x === 11 && s.y === 22 && s.z === 33, 'add sums componentwise' );

  const m = ml.Vector3.mulScalar( { x: 1, y: -2, z: 3 }, 2 );
  assert.ok( m.x === 2 && m.y === -4 && m.z === 6, 'mulScalar scales componentwise' );
}
