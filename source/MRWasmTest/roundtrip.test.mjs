import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const moduleUrl = process.env.MESHLIB_MODULE
  ? pathToFileURL( process.env.MESHLIB_MODULE ).href
  : new URL( './meshlib.mjs', import.meta.url ).href;

const { default: createMeshLib } = await import( moduleUrl );
const ml = await createMeshLib();

function meshFromGeometry( positions, indices ) {
  const coords = ml.VertCoords.fromArray( positions );
  const tris = ml.Triangulation.fromArray( indices );
  const m = ml.Mesh.fromTriangles( coords, tris );
  coords.delete();
  tris.delete();
  return m;
}

function meshToGeometry( mesh, wantNormals ) {
  mesh.pack();
  const points = mesh.points;
  const topology = mesh.topology;
  const tris = topology.getTriangulation();
  const out = { positions: points.toArray(), indices: tris.toArray() };
  points.delete();
  topology.delete();
  tris.delete();
  if ( wantNormals ) {
    const vertNormals = ml.computePerVertNormals( mesh );
    out.normals = vertNormals.toArray();
    vertNormals.delete();
  }
  return out;
}

function cube( cx, cy, cz, s = 2 ) {
  const h = s / 2;
  const positions = new Float32Array( [
    cx - h, cy - h, cz - h,
    cx + h, cy - h, cz - h,
    cx + h, cy + h, cz - h,
    cx - h, cy + h, cz - h,
    cx - h, cy - h, cz + h,
    cx + h, cy - h, cz + h,
    cx + h, cy + h, cz + h,
    cx - h, cy + h, cz + h,
  ] );
  const indices = new Uint32Array( [
    0, 2, 1,  0, 3, 2,
    4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,
    3, 6, 2,  3, 7, 6,
    0, 4, 7,  0, 7, 3,
    1, 2, 6,  1, 6, 5,
  ] );
  return { positions, indices };
}

function isClosedGenus0( verts, tris ) {
  return tris % 2 === 0 && 2 * verts - tris === 4;
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const g = meshToGeometry( m, false );
  assert.ok( g.positions instanceof Float32Array, 'positions is a Float32Array' );
  assert.ok( g.indices instanceof Uint32Array, 'indices is a Uint32Array' );
  assert.equal( g.positions.length, 8 * 3, 'cube welds to 8 vertices' );
  assert.equal( g.indices.length, 12 * 3, 'cube has 12 triangles' );

  const nv = g.positions.length / 3;
  for ( let i = 0; i < g.indices.length; i++ )
    assert.ok( g.indices[i] < nv, 'every index is in range' );
  for ( let f = 0; f < g.indices.length; f += 3 ) {
    const [a, b, d] = [g.indices[f], g.indices[f + 1], g.indices[f + 2]];
    assert.ok( a !== b && b !== d && a !== d, 'no degenerate triangle' );
  }

  let lo = Infinity, hi = -Infinity;
  for ( const v of g.positions ) { lo = Math.min( lo, v ); hi = Math.max( hi, v ); }
  assert.ok( Math.abs( lo + 1 ) < 1e-5 && Math.abs( hi - 1 ) < 1e-5, 'bounding box preserved at [-1,1]' );

  m.delete();
}

{
  const A = cube( 0, 0, 0, 2 );
  const B = cube( 1, 1, 1, 2 );
  const ma = meshFromGeometry( A.positions, A.indices );
  const mb = meshFromGeometry( B.positions, B.indices );

  const res = ml.boolean( ma, mb, ml.BooleanOperation.Union );
  assert.ok( res.valid(), 'union succeeded' );
  const u = res.mesh;

  const gBefore = meshToGeometry( u, false );
  const vBefore = gBefore.positions.length / 3;
  const fBefore = gBefore.indices.length / 3;
  assert.ok( fBefore > 12, 'union has more triangles than a single cube' );
  assert.ok( isClosedGenus0( vBefore, fBefore ),
    `union should be a closed genus-0 solid (V=${vBefore}, F=${fBefore})` );

  const s = new ml.DecimateSettings();
  s.maxDeletedFaces = Math.floor( fBefore / 2 );
  const dr = ml.decimateMesh( u, s );
  assert.ok( dr.facesDeleted >= 0, 'facesDeleted is non-negative' );
  assert.ok( dr.facesDeleted <= Math.floor( fBefore / 2 ), 'decimate honors maxDeletedFaces' );

  const gAfter = meshToGeometry( u, false );
  const fAfter = gAfter.indices.length / 3;
  const vAfter = gAfter.positions.length / 3;
  assert.ok( fAfter <= fBefore, 'decimate never increases triangle count' );
  assert.ok( fAfter >= 4, 'decimate does not destroy the mesh' );
  for ( let i = 0; i < gAfter.indices.length; i++ )
    assert.ok( gAfter.indices[i] < vAfter, 'all indices in range after decimate' );

  const gn = meshToGeometry( u, true );
  assert.ok( gn.normals instanceof Float32Array && gn.normals.length === gn.positions.length,
    'normals array parallels positions' );

  s.delete(); u.delete(); res.delete(); ma.delete(); mb.delete();
}

{
  const A = cube( 0, 0, 0, 2 );
  const Far = cube( 100, 100, 100, 2 );
  const ma = meshFromGeometry( A.positions, A.indices );
  const mf = meshFromGeometry( Far.positions, Far.indices );

  const res = ml.boolean( ma, mf, ml.BooleanOperation.Intersection );
  assert.ok( res.valid(), 'disjoint intersection is still valid' );
  const x = res.mesh;
  assert.equal( meshToGeometry( x, false ).indices.length, 0, 'intersection of disjoint meshes is empty' );

  x.delete(); res.delete(); ma.delete(); mf.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const s = new ml.SelfIntersectionsSettings();
  let calls = 0;
  s.callback = ( progress ) => { calls++; return true; };
  ml.fixSelfIntersections( m, s );
  assert.ok( calls >= 1, 'fix invoked the JS progress callback' );

  const s2 = new ml.SelfIntersectionsSettings();
  s2.callback = () => false;
  assert.throws( () => ml.fixSelfIntersections( m, s2 ), /cancel/i,
    'a cancelled fix surfaces the Expected<void> error as a JS exception' );

  s.delete(); s2.delete(); m.delete();
}

console.log( 'OK' );
