import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry, meshToGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const eq = ( a, b ) => Math.abs( a - b ) < 1e-4;

  const bb = m.computeBoundingBox();
  assert.ok( bb.valid(), 'computed bounding box is valid' );
  assert.ok( eq( bb.min.x, -1 ) && eq( bb.min.y, -1 ) && eq( bb.min.z, -1 ), 'bbox min at (-1,-1,-1)' );
  assert.ok( eq( bb.max.x, 1 ) && eq( bb.max.y, 1 ) && eq( bb.max.z, 1 ), 'bbox max at (1,1,1)' );
  const sz = bb.size();
  assert.ok( eq( sz.x, 2 ) && eq( sz.y, 2 ) && eq( sz.z, 2 ), 'bbox size is (2,2,2)' );
  const ctr = bb.center();
  assert.ok( eq( ctr.x, 0 ) && eq( ctr.y, 0 ) && eq( ctr.z, 0 ), 'bbox center at origin' );
  assert.ok( eq( bb.volume(), 8 ), 'bbox volume is 8' );
  bb.delete();

  assert.ok( eq( m.volume(), 8 ), 'cube volume is 8' );
  assert.ok( eq( m.area(), 24 ), 'cube surface area is 24' );
  m.invalidateCaches();

  m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const eq = ( a, b ) => Math.abs( a - b ) < 1e-4;

  const avg = m.averageEdgeLength();
  assert.ok( avg > 1.9 && avg < 2.9, 'average edge length is between the cube edge (2) and face diagonal (~2.83)' );

  for ( const ctr of [ m.findCenterFromPoints(), m.findCenterFromFaces(), m.findCenterFromBBox() ] )
    assert.ok( eq( ctr.x, 0 ) && eq( ctr.y, 0 ) && eq( ctr.z, 0 ), 'a cube at the origin is centered at (0,0,0)' );

  const xf = ml.AffineXf3f.translation( { x: 5, y: 0, z: 0 } );
  m.transform( xf );
  xf.delete();

  for ( const ctr of [ m.findCenterFromPoints(), m.findCenterFromFaces(), m.findCenterFromBBox() ] )
    assert.ok( eq( ctr.x, 5 ) && eq( ctr.y, 0 ) && eq( ctr.z, 0 ), 'after translating +5x every center moves to (5,0,0)' );

  const bb = m.computeBoundingBox();
  assert.ok( eq( bb.min.x, 4 ) && eq( bb.max.x, 6 ), 'bounding box shifted +5 in x' );
  bb.delete();

  m.delete();
}

{
  const ca = cube( 0, 0, 0, 2 );
  const a = meshFromGeometry( ca.positions, ca.indices );
  const cb = cube( 10, 0, 0, 2 );
  const b = meshFromGeometry( cb.positions, cb.indices );

  a.addMesh( b );

  const g = meshToGeometry( a, false );
  assert.equal( g.positions.length, 16 * 3, 'combined mesh has 16 vertices' );
  assert.equal( g.indices.length, 24 * 3, 'combined mesh has 24 triangles' );
  assert.ok( Math.abs( a.volume() - 16 ) < 1e-4, 'two size-2 cubes total volume 16' );

  a.delete();
  b.delete();
}
