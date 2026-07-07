import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const a = cube( 0, 0, 0, 2 );
  const b = cube( 10, 0, 0, 2 );
  const positions = new Float32Array( [ ...a.positions, ...b.positions ] );
  const indices = new Uint32Array( [ ...a.indices, ...Array.from( b.indices, i => i + 8 ) ] );
  const m = meshFromGeometry( positions, indices );
  const PerEdge = ml.MeshComponentsFaceIncidence.PerEdge;

  assert.equal( ml.MeshComponents.getNumComponents( m, PerEdge ), 2, 'two disjoint cubes are two components' );

  const largest = ml.MeshComponents.getLargestComponent( m, PerEdge, 0 );
  assert.ok( largest instanceof ml.FaceBitSet, 'getLargestComponent returns a FaceBitSet' );
  assert.equal( largest.count(), 12, 'the largest component is one whole cube (12 faces)' );

  const containingFace0 = ml.MeshComponents.getComponent( m, 0, PerEdge );
  assert.equal( containingFace0.count(), 12, 'the component containing face 0 is one cube' );

  const bothSeeds = ml.FaceBitSet.fromIndices( new Uint32Array( [ 0, 12 ] ) );
  const both = ml.MeshComponents.getComponents( m, bothSeeds, PerEdge );
  assert.equal( both.count(), 24, 'seeding one face in each cube selects all 24 faces' );

  const allByArea = ml.MeshComponents.getLargeByAreaComponents( m, 0.1 );
  assert.equal( allByArea.count(), 24, 'both cubes exceed a tiny area threshold' );
  const noneByArea = ml.MeshComponents.getLargeByAreaComponents( m, 1e9 );
  assert.equal( noneByArea.count(), 0, 'no component is larger than a huge area threshold' );

  largest.delete();
  containingFace0.delete();
  bothSeeds.delete();
  both.delete();
  allByArea.delete();
  noneByArea.delete();
  m.delete();
}
