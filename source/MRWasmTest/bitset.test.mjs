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
