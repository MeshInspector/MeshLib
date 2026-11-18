import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

// Vector<T,Id> map round-trips
{
  const fm = ml.FaceMap.fromArray( new Uint32Array( [ 3, 1, 0, 2 ] ) );
  assert.deepEqual( Array.from( fm.toArray() ), [ 3, 1, 0, 2 ], 'FaceMap round-trips' );
  fm.delete();

  const vm = ml.VertMap.fromArray( new Uint32Array( [ 5, 4 ] ) );
  assert.deepEqual( Array.from( vm.toArray() ), [ 5, 4 ], 'VertMap round-trips' );
  vm.delete();
}

// two disjoint cubes → component maps
{
  const a = cube( 0, 0, 0, 2 ), b = cube( 10, 0, 0, 2 );
  const positions = new Float32Array( [ ...a.positions, ...b.positions ] );
  const indices = new Uint32Array( [ ...a.indices, ...Array.from( b.indices, i => i + 8 ) ] );
  const m = meshFromGeometry( positions, indices );
  const PerEdge = ml.MeshComponentsFaceIncidence.PerEdge;

  const comps = ml.MeshComponents.getAllComponents( m, PerEdge );
  assert.equal( comps.length, 2, 'getAllComponents returns two components' );
  assert.ok( comps.every( c => c instanceof ml.FaceBitSet && c.count() === 12 ), 'each component is a 12-face cube' );
  comps.forEach( c => c.delete() );

  const cm = ml.MeshComponents.getAllComponentsMap( m, PerEdge );
  assert.equal( cm.numRegions, 2, 'two regions' );
  assert.ok( cm.map instanceof ml.Face2RegionMap, 'map is a Face2RegionMap' );
  assert.equal( cm.map.toArray().length, 24, 'one region id per face' );

  const big = ml.MeshComponents.getLargeByAreaRegions( m, cm.map, cm.numRegions, 0.1 );
  assert.equal( big.faces.count(), 24, 'both regions exceed a tiny area' );
  big.faces.delete();
  const none = ml.MeshComponents.getLargeByAreaRegions( m, cm.map, cm.numRegions, 1e9 );
  assert.equal( none.faces.count(), 0, 'no region is that large' );
  none.faces.delete();

  cm.map.delete();
  m.delete();
}

// uniteCloseVertices — a clean cube has none within a tiny distance
{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  assert.equal( ml.MeshBuilder.uniteCloseVertices( m, 0.001, true ), 0, 'no close vertices to unite' );
  m.delete();
}
