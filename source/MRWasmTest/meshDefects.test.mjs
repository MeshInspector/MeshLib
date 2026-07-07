import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const shortEdges = ml.findShortEdges( m, 10 );
  assert.ok( shortEdges instanceof ml.UndirectedEdgeBitSet, 'findShortEdges returns an UndirectedEdgeBitSet' );
  assert.equal( shortEdges.count(), 18, 'a closed 12-triangle cube has 18 undirected edges, all shorter than 10' );
  shortEdges.delete();

  const noShortEdges = ml.findShortEdges( m, 0.01 );
  assert.equal( noShortEdges.count(), 0, 'no edge is shorter than 0.01' );
  noShortEdges.delete();

  const degenerate = ml.findDegenerateFaces( m, 1e6 );
  assert.ok( degenerate instanceof ml.FaceBitSet, 'findDegenerateFaces returns a FaceBitSet' );
  assert.equal( degenerate.count(), 0, 'a clean cube has no faces above a huge aspect ratio' );
  degenerate.delete();

  const holeFaces = ml.findHoleComplicatingFaces( m );
  assert.equal( holeFaces.count(), 0, 'a closed cube has no hole-complicating faces' );
  holeFaces.delete();

  m.delete();
}
