import assert from 'node:assert/strict';
import { ml, meshToGeometry } from './helpers.mjs';

function positionsChanged( before, after ) {
  for ( let i = 0; i < before.length; ++i )
    if ( Math.abs( before[ i ] - after[ i ] ) > 1e-6 )
      return true;
  return false;
}

// FreeFormDeformer: moving a control-grid corner deforms the mesh
{
  const mesh = ml.makeUVSphere( 1, 24, 24 );
  const before = meshToGeometry( mesh, false ).positions;

  const ffd = new ml.FreeFormDeformer( mesh );
  ffd.init( { x: 2, y: 2, z: 2 } );  // 2x2x2 control grid over the mesh bounding box

  const corner = { x: 0, y: 0, z: 0 };
  const p = ffd.getRefGridPointPosition( corner );
  ffd.setRefGridPointPosition( corner, { x: p.x - 1, y: p.y - 1, z: p.z - 1 } );
  ffd.apply();

  const after = meshToGeometry( mesh, false ).positions;
  assert.equal( before.length, after.length, 'vertex count is unchanged' );
  assert.ok( positionsChanged( before, after ), 'free-form deformation moved mesh vertices' );

  ffd.delete();
  mesh.delete();
}

// Laplacian: fixing a freed vertex at a new position smoothly deforms the region
{
  const mesh = ml.makeUVSphere( 1, 24, 24 );
  const before = meshToGeometry( mesh, false ).positions;

  const lap = new ml.Laplacian( mesh );

  // free a small patch (top cap) — must not be the whole connected component
  const freeIdx = new Uint32Array( 24 );
  for ( let i = 0; i < freeIdx.length; ++i )
    freeIdx[ i ] = i;
  const freeVerts = ml.VertBitSet.fromIndices( freeIdx );

  lap.init( freeVerts, ml.EdgeWeights.Cotan );  // 2-arg form: vmass=Unit, rem=Yes
  lap.fixVertex( 0, { x: 0, y: 0, z: 2 }, true );  // 3-arg form: pull vertex 0 outward
  lap.apply();

  const after = meshToGeometry( mesh, false ).positions;
  assert.ok( positionsChanged( before, after ), 'laplacian deformation moved mesh vertices' );

  freeVerts.delete();
  lap.delete();
  mesh.delete();
}
