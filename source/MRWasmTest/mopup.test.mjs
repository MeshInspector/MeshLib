import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

// makeConvexHullFromMesh / makeConvexHullFromPoints
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const hullM = ml.makeConvexHullFromMesh( m );
  assert.ok( meshToGeometry( hullM, false ).positions.length > 0, 'convex hull from mesh is non-empty' );

  const pc = ml.meshToPointCloud( m, false );
  const hullP = ml.makeConvexHullFromPoints( pc );
  assert.ok( meshToGeometry( hullP, false ).positions.length > 0, 'convex hull from points is non-empty' );

  hullM.delete();
  hullP.delete();
  pc.delete();
  m.delete();
}

// expandVerts / shrinkVerts (hops)
{
  const sphere = ml.makeUVSphere( 1, 16, 16 );
  const topo = sphere.topology;
  const seed = ml.VertBitSet.fromIndices( new Uint32Array( [ 0 ] ) );

  const expanded = ml.expandVerts( topo, seed, 1 );
  assert.ok( expanded.count() > 1, 'expandVerts grows the region beyond the seed' );
  const shrunk = ml.shrinkVerts( topo, expanded, 1 );
  assert.ok( shrunk.count() < expanded.count(), 'shrinkVerts reduces the region' );

  seed.delete();
  expanded.delete();
  shrunk.delete();
  topo.delete();
  sphere.delete();
}

// findRightBoundary / trackRightBoundaryLoop on an open mesh
{
  const m = meshFromGeometry( new Float32Array( [ 0, 0, 0, 1, 0, 0, 0, 1, 0 ] ), new Uint32Array( [ 0, 1, 2 ] ) );
  const topo = m.topology;

  const loops = ml.findRightBoundary( topo );
  assert.ok( loops.length >= 1, 'open mesh has a boundary loop' );
  assert.ok( loops[ 0 ].length >= 3, 'triangle boundary loop has at least 3 edges' );

  const loop = ml.trackRightBoundaryLoop( topo, loops[ 0 ][ 0 ] );
  assert.ok( loop.length >= 3, 'tracked boundary loop has at least 3 edges' );

  topo.delete();
  m.delete();
}

// VertColors round-trip (RGBA bytes)
{
  const rgba = new Uint8Array( [ 255, 0, 0, 255, 0, 255, 0, 255 ] );
  const vc = ml.VertColors.fromArray( rgba );
  const out = vc.toArray();
  assert.equal( out.length, rgba.length, 'VertColors round-trips length' );
  assert.equal( out[ 0 ], 255, 'first color red channel preserved' );
  assert.equal( out[ 5 ], 255, 'second color green channel preserved' );
  vc.delete();
}

// fillHoleNicely closes a triangular hole (open tetrahedron)
{
  const m = meshFromGeometry(
    new Float32Array( [ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.3, 0.3, 1 ] ),
    new Uint32Array( [ 0, 1, 3, 1, 2, 3, 2, 0, 3 ] ) );
  const topo = m.topology;
  const holeEdges = topo.findHoleRepresentiveEdges();
  assert.ok( holeEdges.length >= 1, 'open tetrahedron has a hole' );

  const settings = new ml.FillHoleNicelySettings();
  settings.triangulateOnly = true;
  const patch = ml.fillHoleNicely( m, holeEdges[ 0 ], settings );
  assert.ok( patch.count() >= 1, 'fillHoleNicely produced patch faces' );

  patch.delete();
  settings.delete();
  topo.delete();
  m.delete();
}
