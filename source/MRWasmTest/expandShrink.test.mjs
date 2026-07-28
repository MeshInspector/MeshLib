import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const topo = m.topology;
  const all = topo.getValidFaces();
  const one = ml.FaceBitSet.fromIndices( new Uint32Array( [ 0 ] ) );

  const expandedOne = ml.expandFaces( topo, one );
  assert.ok( expandedOne instanceof ml.FaceBitSet, 'expandFaces returns a FaceBitSet' );
  assert.equal( expandedOne.count(), 4, 'a triangle plus its 3 edge-neighbors' );
  assert.ok( expandedOne.test( 0 ), 'expansion keeps the original face' );

  const expandedAll = ml.expandFaces( topo, all );
  assert.equal( expandedAll.count(), 12, 'expanding the whole mesh cannot exceed it' );

  const shrunkAll = ml.shrinkFaces( topo, all );
  assert.equal( shrunkAll.count(), 12, 'a closed mesh has no boundary, so the full region does not shrink' );

  const shrunkOne = ml.shrinkFaces( topo, one );
  assert.equal( shrunkOne.count(), 0, 'a lone face is entirely region-boundary, so it is removed' );

  const bdOne = ml.getBoundaryFaces( topo, one );
  assert.equal( bdOne.count(), 1, 'the lone face is its own region boundary' );

  const bdAll = ml.getBoundaryFaces( topo, all );
  assert.equal( bdAll.count(), 0, 'a closed full region has no boundary faces' );

  all.delete();
  one.delete();
  expandedOne.delete();
  expandedAll.delete();
  shrunkAll.delete();
  shrunkOne.delete();
  bdOne.delete();
  bdAll.delete();
  topo.delete();
  m.delete();
}
