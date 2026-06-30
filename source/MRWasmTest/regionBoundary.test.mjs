import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );
  const topo = m.topology;
  const allFaces = topo.getValidFaces();   // 12
  const allVerts = topo.getValidVerts();   // 8
  const face0 = ml.FaceBitSet.fromIndices( new Uint32Array( [ 0 ] ) );
  const vert0 = ml.VertBitSet.fromIndices( new Uint32Array( [ 0 ] ) );

  const v = ml.getIncidentVertsFromFaces( topo, face0 );
  assert.equal( v.count(), 3, 'a triangle has 3 incident vertices' );
  const e = ml.getIncidentEdgesFromFaces( topo, face0 );
  assert.equal( e.count(), 3, 'a triangle has 3 incident edges' );

  const f = ml.getIncidentFacesFromVerts( topo, vert0 );
  assert.equal( f.count(), 6, 'corner vertex 0 is shared by 6 triangles' );

  const ve = ml.getIncidentVertsFromEdges( topo, e );
  assert.equal( ve.count(), 3, "face 0's edges connect its 3 vertices" );
  const fe = ml.getIncidentFacesFromEdges( topo, e );
  assert.ok( fe.count() >= 1, 'faces incident to those edges' );

  const innerF = ml.getInnerFaces( topo, allVerts );
  assert.equal( innerF.count(), 12, 'all faces are inner to all vertices' );
  const innerV = ml.getInnerVertsFromFaces( topo, allFaces );
  assert.equal( innerV.count(), 8, 'all vertices are inner to all faces' );
  const innerEv = ml.getInnerEdgesFromVerts( topo, allVerts );
  assert.ok( innerEv.count() > 0, 'inner edges from all vertices' );
  const innerEf = ml.getInnerEdgesFromFaces( topo, allFaces );
  assert.ok( innerEf.count() > 0, 'inner edges from all faces' );
  const innerVe = ml.getInnerVertsFromEdges( topo, e );
  assert.ok( innerVe.count() >= 0, 'inner verts from edges (smoke)' );

  const bv = ml.getBoundaryVerts( topo, allFaces );
  assert.equal( bv.count(), 0, 'closed cube full region has no boundary vertices' );

  for ( const bs of [ allFaces, allVerts, face0, vert0, v, e, f, ve, fe, innerF, innerV, innerEv, innerEf, innerVe, bv ] )
    bs.delete();
  topo.delete();
  m.delete();
}
