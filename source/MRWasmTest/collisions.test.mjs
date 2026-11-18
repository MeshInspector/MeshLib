import assert from 'node:assert/strict';
import { ml } from './helpers.mjs';

// findCollidingTriangles / findCollidingTriangleBitsets — two overlapping spheres
{
  const a = ml.makeUVSphere( 1, 16, 16 );
  const b = ml.makeUVSphere( 1, 16, 16 );
  const shift = ml.AffineXf3f.translation( { x: 0.5, y: 0, z: 0 } );
  b.transform( shift );

  const pairs = ml.findCollidingTriangles( a, b );
  assert.ok( pairs.length > 0, 'overlapping spheres have colliding triangle pairs' );
  assert.ok( Number.isInteger( pairs[ 0 ].aFace ) && Number.isInteger( pairs[ 0 ].bFace ),
    'each pair carries aFace / bFace face indices' );

  const bitsets = ml.findCollidingTriangleBitsets( a, b );
  assert.ok( bitsets.a.count() > 0 && bitsets.b.count() > 0,
    'bitsets mark colliding faces on both meshes' );
  bitsets.a.delete();
  bitsets.b.delete();

  // fast path: firstIntersectionOnly still reports the collision
  assert.ok( ml.findCollidingTriangles( a, b, true ).length > 0, 'firstIntersectionOnly reports a hit' );

  shift.delete();
  a.delete();
  b.delete();
}

// no collision — far-apart spheres
{
  const a = ml.makeUVSphere( 1, 16, 16 );
  const b = ml.makeUVSphere( 1, 16, 16 );
  const shift = ml.AffineXf3f.translation( { x: 5, y: 0, z: 0 } );
  b.transform( shift );

  assert.equal( ml.findCollidingTriangles( a, b ).length, 0, 'far-apart spheres do not collide' );

  shift.delete();
  a.delete();
  b.delete();
}

// self-collisions — a deliberately self-intersecting torus
{
  const torus = ml.makeTorusWithSelfIntersections( 1, 0.1, 16, 16 );

  const pairs = ml.findSelfCollidingTriangles( torus );
  assert.ok( pairs.length > 0, 'the self-intersecting torus reports self-colliding pairs' );

  const bs = ml.findSelfCollidingTrianglesBS( torus );
  assert.ok( bs.count() > 0, 'the self-intersecting faces form a non-empty set' );
  bs.delete();

  torus.delete();
}

// precise collisions — enumerable VarEdgeTri entries
{
  const a = ml.makeUVSphere( 1, 16, 16 );
  const b = ml.makeUVSphere( 1, 16, 16 );
  const shift = ml.AffineXf3f.translation( { x: 0.5, y: 0, z: 0 } );
  b.transform( shift );

  const conv = ml.getVectorConverters( a, b );
  const res = ml.findCollidingEdgeTrisPrecise( a, b, conv );
  assert.ok( res.size() > 0, 'precise collision finds edge/triangle intersections' );

  const vet = res.get( 0 );
  assert.equal( typeof vet.isEdgeATriB(), 'boolean', 'VarEdgeTri.isEdgeATriB() is a boolean' );
  assert.ok( Number.isInteger( vet.edge ) && Number.isInteger( vet.tri() ), 'VarEdgeTri exposes edge and tri()' );
  vet.delete();
  res.delete();
  conv.delete();

  shift.delete();
  a.delete();
  b.delete();
}
