import assert from 'node:assert/strict';
import { ml, meshToGeometry } from './helpers.mjs';

// VertScalars round-trips a Float32Array
{
  const src = new Float32Array( [ 0.1, 0.2, 0.3, 0.4 ] );
  const scalars = ml.VertScalars.fromArray( src );
  assert.deepEqual( Array.from( scalars.toArray() ), Array.from( src ), 'VertScalars round-trips the weights' );
  scalars.delete();
}

// weighted shell offsets a torus using per-vertex weights
{
  const mesh = ml.makeTorus( 1.0, 0.1, 16, 16 );

  const coords = mesh.points;
  const positions = coords.toArray();
  const weights = new Float32Array( positions.length / 3 );
  let maxWeight = 0;
  for ( let v = 0; v < weights.length; ++v ) {
    weights[ v ] = Math.abs( positions[ 3 * v ] / 5 );
    maxWeight = Math.max( maxWeight, weights[ v ] );
  }
  coords.delete();
  const scalars = ml.VertScalars.fromArray( weights );

  const params = new ml.WeightedShellParametersMetric();
  params.voxelSize = ml.suggestVoxelSize( mesh, 1000 );
  params.offset = 0.2;
  const dist = params.dist;
  dist.maxWeight = maxWeight;
  params.dist = dist;
  dist.delete();

  const result = ml.WeightedShell.meshShell( mesh, scalars, params );
  const g = meshToGeometry( result, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'weighted shell produces a non-empty mesh' );

  result.delete();
  params.delete();
  scalars.delete();
  mesh.delete();
}
