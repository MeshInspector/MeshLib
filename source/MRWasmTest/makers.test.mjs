import assert from 'node:assert/strict';
import { ml, meshToGeometry } from './helpers.mjs';

{
  const m = ml.makeCube( { x: 2, y: 2, z: 2 }, { x: -1, y: -1, z: -1 } );
  const g = meshToGeometry( m, false );
  assert.equal( g.positions.length, 8 * 3, 'cube has 8 vertices' );
  assert.equal( g.indices.length, 12 * 3, 'cube has 12 triangles' );
  assert.ok( Math.abs( m.volume() - 8 ) < 1e-4, 'size-2 cube has volume 8' );
  m.delete();
}

{
  const m = ml.makeUVSphere( 1, 16, 16 );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'uv sphere is non-empty' );
  const bb = m.computeBoundingBox();
  const sz = bb.size();
  assert.ok( sz.x > 1.9 && sz.x <= 2.001 && sz.z > 1.9 && sz.z <= 2.001, 'uv sphere spans ~2 (unit radius)' );
  bb.delete();
  m.delete();
}

{
  const m = ml.makeSphere( { radius: 1, numMeshVertices: 100 } );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'irregular sphere is non-empty' );
  m.delete();
}

{
  const m = ml.makeCylinderAdvanced( 0.1, 0.1, 0, 2 * Math.PI, 1, 16 );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'cylinder is non-empty' );
  m.delete();
}

{
  const m = ml.makeTorus( 1, 0.1, 16, 16 );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'torus is non-empty' );
  m.delete();
}

{
  const m = ml.makeTorusWithSelfIntersections( 1, 0.1, 16, 16 );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'self-intersecting torus is non-empty' );
  m.delete();
}

{
  const c = ml.makeCube( { x: 2, y: 2, z: 2 }, { x: -1, y: -1, z: -1 } );
  const hull = ml.makeConvexHullFromMesh( c );
  const g = meshToGeometry( hull, false );
  assert.equal( g.positions.length, 8 * 3, 'convex hull of a box is the box (8 vertices)' );
  c.delete();
  hull.delete();
}
