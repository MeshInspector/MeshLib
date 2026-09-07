import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const topo = m.topology;

  const faces = topo.getValidFaces();
  assert.equal( faces.count(), 12, 'cube has 12 valid faces' );
  assert.equal( faces.size(), 12, 'face bitset is sized to the face count' );
  assert.ok( !faces.empty(), 'valid faces are non-empty' );
  assert.ok( faces.test( 0 ), 'face 0 is valid' );

  const faceIdx = faces.toIndices();
  assert.equal( faceIdx.length, 12, 'toIndices lists all 12 faces' );
  assert.equal( faceIdx[ 0 ], 0 );

  const verts = topo.getValidVerts();
  assert.equal( verts.count(), 8, 'cube has 8 valid vertices' );

  faces.delete();
  verts.delete();
  topo.delete();
  m.delete();
}

{
  const fbs = ml.FaceBitSet.fromIndices( new Uint32Array( [ 1, 3, 5 ] ) );
  assert.equal( fbs.count(), 3, 'fromIndices sets 3 bits' );
  assert.equal( fbs.size(), 6, 'fromIndices sizes to the max index + 1' );
  assert.ok( fbs.test( 5 ) && !fbs.test( 4 ), 'the correct bits are set' );
  assert.deepEqual( Array.from( fbs.toIndices() ), [ 1, 3, 5 ], 'round-trips the indices' );
  fbs.delete();
}

// set / resize / find_first / find_last
{
  const bs = new ml.VertBitSet();
  bs.resize( 10, false );
  assert.equal( bs.size(), 10, 'resize sets the size' );
  assert.equal( bs.count(), 0, 'resize(…, false) leaves every bit clear' );

  bs.set( 3, true );
  bs.set( 7, true );
  assert.equal( bs.count(), 2, 'two bits set' );
  assert.equal( bs.find_first(), 3, 'find_first returns the lowest set bit' );
  assert.equal( bs.find_last(), 7, 'find_last returns the highest set bit' );

  bs.set( 3, false );
  assert.equal( bs.find_first(), 7, 'clearing bit 3 advances find_first to 7' );

  bs.delete();
}
