import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const topo = m.topology;
  const eq = ( a, b ) => Math.abs( a - b ) < 1e-4;

  assert.equal( topo.faceSize(), 12, 'a cube has 12 faces' );
  assert.equal( topo.findNumHoles(), 0, 'a closed cube has no holes' );
  assert.equal( topo.findHoleRepresentiveEdges().length, 0, 'a closed cube has no hole edges' );

  const tri = topo.getTriVerts( 0 );
  assert.equal( tri.length, 3, 'a triangle has 3 vertices' );
  assert.ok( tri.every( v => Number.isInteger( v ) && v >= 0 && v < 8 ), 'vertices are valid indices' );
  assert.ok( tri[ 0 ] !== tri[ 1 ] && tri[ 1 ] !== tri[ 2 ] && tri[ 0 ] !== tri[ 2 ], 'vertices are distinct' );

  const left = topo.getLeftTriVerts( 0 );
  assert.equal( left.length, 3, 'getLeftTriVerts returns 3 vertices' );

  const len = m.edgeLength( 0 );
  assert.ok( len > 1.9 && len < 2.9, 'a cube edge is 2 (edge) or ~2.83 (face diagonal)' );
  assert.ok( eq( m.edgeLengthSq( 0 ), len * len ), 'edgeLengthSq is the square of edgeLength' );

  const tp = m.toTriPoint( 0, { x: 0, y: 0, z: -1 } );
  assert.ok( Number.isInteger( tp.e ), 'toTriPoint edge is an int' );
  assert.ok( typeof tp.bary.a === 'number' && typeof tp.bary.b === 'number', 'barycentric coords present' );

  topo.delete();
  m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const coords = ml.VertCoords.fromArray( c.positions );
  const tris = ml.Triangulation.fromArray( c.indices );
  const m = ml.Mesh.fromTrianglesDuplicatingNonManifoldVertices( coords, tris );
  coords.delete();
  tris.delete();

  const topo = m.topology;
  assert.equal( topo.faceSize(), 12, 'a manifold cube needs no duplication (12 faces)' );
  topo.delete();
  m.delete();
}

// Mesh.normal and VertCoords.get / size
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const points = m.points;

  assert.equal( points.size(), 8, 'a size-2 cube has 8 vertices' );
  const p0 = points.get( 0 );
  assert.ok( Math.abs( p0.x + 1 ) < 1e-4 && Math.abs( p0.y + 1 ) < 1e-4 && Math.abs( p0.z + 1 ) < 1e-4,
    'vertex 0 is the (-1,-1,-1) corner' );

  const n = m.normal( 0 );
  assert.ok( Math.abs( Math.hypot( n.x, n.y, n.z ) - 1 ) < 1e-3, 'the vertex normal is unit length' );

  points.delete();
  m.delete();
}
