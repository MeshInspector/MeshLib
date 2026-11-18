import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const coordsBefore = m.points;
  const beforeVerts = coordsBefore.size();
  coordsBefore.delete();

  // a degenerate band duplicates the boundary vertices of the extruded region
  const faces = ml.FaceBitSet.fromIndices( new Uint32Array( [ 1, 2 ] ) );
  ml.makeDegenerateBandAroundRegion( m, faces );

  const coords = m.points;
  assert.ok( coords.size() > beforeVerts, 'the band adds duplicated boundary vertices' );

  // shift the incident vertices via the writable points accessors
  const topo = m.topology;
  const verts = ml.getIncidentVertsFromFaces( topo, faces );
  const idx = verts.toIndices();
  assert.ok( idx.length > 0, 'the extruded faces have incident vertices' );

  const v0 = idx[ 0 ];
  const zBefore = coords.get( v0 ).z;
  for ( const v of idx ) {
    const p = coords.get( v );
    p.z += 1;
    coords.set( v, p );
  }
  m.points = coords;
  m.invalidateCaches();

  const after = m.points;
  assert.ok( Math.abs( after.get( v0 ).z - ( zBefore + 1 ) ) < 1e-5, 'the moved vertex is written back to the mesh' );

  after.delete();
  coords.delete();
  verts.delete();
  topo.delete();
  faces.delete();
  m.delete();
}
