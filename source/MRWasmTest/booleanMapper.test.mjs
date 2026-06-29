import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const ca = cube( 0, 0, 0, 2 ), cb = cube( 1, 0, 0, 2 ); // overlapping along x
  const a = meshFromGeometry( ca.positions, ca.indices );
  const b = meshFromGeometry( cb.positions, cb.indices );

  const mapper = new ml.BooleanResultMapper();
  const res = ml.boolean( a, b, ml.BooleanOperation.Union, mapper );
  assert.ok( res.valid(), 'union of overlapping cubes succeeds' );

  const newF = mapper.newFaces();
  assert.ok( newF instanceof ml.FaceBitSet && newF.count() > 0, 'the cut introduces new faces' );

  const topoA = a.topology;
  const aFaces = topoA.getValidFaces();
  const mappedA = mapper.mapFaces( aFaces, ml.BooleanMapObject.A );
  assert.ok( mappedA instanceof ml.FaceBitSet && mappedA.count() > 0, 'A faces map into the result' );

  const n2o = mapper.getNew2OldFaceMap( ml.BooleanMapObject.A );
  assert.ok( n2o instanceof ml.FaceMap, 'getNew2OldFaceMap returns a FaceMap' );

  newF.delete();
  aFaces.delete();
  mappedA.delete();
  n2o.delete();
  topoA.delete();
  res.delete();
  mapper.delete();
  a.delete();
  b.delete();
}
