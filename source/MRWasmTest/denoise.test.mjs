import assert from 'node:assert/strict';
import { ml, meshToGeometry } from './helpers.mjs';

// addNoise perturbs the vertices; meshDenoiseViaNormals then smooths them
{
  const sphere = ml.makeUVSphere( 1, 16, 16 );

  const before = meshToGeometry( sphere, false ).positions;

  const noise = new ml.NoiseSettings();
  noise.sigma = 0.01;
  noise.seed = 1;
  ml.addNoise( sphere, noise );
  sphere.invalidateCaches();
  noise.delete();

  const noised = meshToGeometry( sphere, false ).positions;
  assert.equal( noised.length, before.length, 'noise keeps the vertex count' );
  let moved = 0;
  for ( let i = 0; i < before.length; ++i )
    if ( Math.abs( before[ i ] - noised[ i ] ) > 1e-6 ) ++moved;
  assert.ok( moved > 0, 'addNoise perturbs the vertices' );

  ml.meshDenoiseViaNormals( sphere );
  assert.ok( meshToGeometry( sphere, false ).positions.length > 0, 'the mesh survives denoising' );

  sphere.delete();
}
