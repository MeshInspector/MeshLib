import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

const eq = ( a, b ) => Math.abs( a - b ) < 1e-3;

// findDistance — two cubes separated along x
{
  const ca = cube( 0, 0, 0, 2 ), cb = cube( 5, 0, 0, 2 );
  const a = meshFromGeometry( ca.positions, ca.indices );
  const b = meshFromGeometry( cb.positions, cb.indices );

  const r = ml.findDistance( a, b );
  assert.ok( eq( r.distSq, 9 ), 'gap between [-1,1] and [4,6] cubes is 3 (distSq 9)' );
  assert.ok( eq( r.a.point.x, 1 ) && eq( r.b.point.x, 4 ), 'closest points lie on the facing faces' );
  assert.ok( Number.isInteger( r.a.face ) && Number.isInteger( r.b.face ), 'face indices present' );

  a.delete();
  b.delete();
}

// isInside — small cube within big cube
{
  const cs = cube( 0, 0, 0, 1 ), cb = cube( 0, 0, 0, 4 );
  const small = meshFromGeometry( cs.positions, cs.indices );
  const big = meshFromGeometry( cb.positions, cb.indices );

  assert.equal( ml.isInside( small, big ), true, 'small cube is inside big cube' );
  assert.equal( ml.isInside( big, small ), false, 'big cube is not inside small cube' );

  small.delete();
  big.delete();
}

// findSignedDistanceFromPoint — sign flips inside vs outside
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const outside = ml.findSignedDistanceFromPoint( { x: 5, y: 0, z: 0 }, m );
  assert.notEqual( outside, null, 'a projection exists' );
  assert.ok( eq( outside.dist, 4 ), 'point (5,0,0) is 4 outside the +x face' );
  assert.ok( eq( outside.proj.point.x, 1 ), 'closest point on the +x face' );

  const inside = ml.findSignedDistanceFromPoint( { x: 0, y: 0, z: 0 }, m );
  assert.ok( inside.dist < 0 && eq( inside.dist, -1 ), 'the centre is 1 unit inside (negative distance)' );

  m.delete();
}

// findSignedDistanceFromMesh — signed distance between two separated cubes
{
  const ca = cube( 0, 0, 0, 2 ), cb = cube( 5, 0, 0, 2 );
  const a = meshFromGeometry( ca.positions, ca.indices );
  const b = meshFromGeometry( cb.positions, cb.indices );

  const r = ml.findSignedDistanceFromMesh( a, b );
  assert.ok( eq( r.signedDist, 3 ), 'the [-1,1] and [4,6] cubes are 3 apart (positive: not colliding)' );
  assert.ok( Number.isInteger( r.status ), 'a MeshMeshCollisionStatus value is returned' );

  a.delete();
  b.delete();
}

// findSignedDistances — verts of a mesh lie on its own surface
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const params = new ml.MeshProjectionParameters();
  const d = ml.findSignedDistances( m, m, params );
  assert.equal( d.length, 8, 'one signed distance per cube vertex' );
  assert.ok( d.every( v => Math.abs( v ) < 1e-3 ), 'every vertex lies on the reference surface (~0)' );

  params.delete();
  m.delete();
}
