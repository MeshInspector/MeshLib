import assert from 'node:assert/strict';
import { ml } from './helpers.mjs';

// single-pair ICP: align a translated copy of a sphere back onto itself
{
  const sphere = ml.makeUVSphere( 1, 32, 32 );

  const refMop = ml.MeshOrPoints.fromMesh( sphere );
  const fltMop = ml.MeshOrPoints.fromMesh( sphere );

  const refXf = new ml.AffineXf3f();                                  // identity
  const fltXf = ml.AffineXf3f.translation( { x: 0.2, y: 0, z: 0 } );  // initial offset

  const ref = new ml.MeshOrPointsXf( refMop, refXf );
  const flt = new ml.MeshOrPointsXf( fltMop, fltXf );

  const icp = new ml.ICP( flt, ref, 0.1 );  // 3-arg form: automatic voxel sampling
  icp.updatePointPairs();                    // pairs' active set is sized only after an update
  assert.ok( icp.getNumSamples() > 0, 'ICP formed sample pairs' );

  const xf = icp.calculateTransformation();
  const meanSq = icp.getMeanSqDistToPoint();
  assert.ok( meanSq < 0.05, `ICP converged (RMS dist to point = ${meanSq})` );

  xf.delete();
  icp.delete();
  flt.delete(); ref.delete();
  fltXf.delete(); refXf.delete();
  fltMop.delete(); refMop.delete();
  sphere.delete();
}

// ICPProperties round-trips and drives setParams (point-to-point method)
{
  const sphere = ml.makeUVSphere( 1, 24, 24 );
  const aMop = ml.MeshOrPoints.fromMesh( sphere );
  const bMop = ml.MeshOrPoints.fromMesh( sphere );
  const aXf = new ml.AffineXf3f();
  const bXf = ml.AffineXf3f.translation( { x: 0.1, y: 0, z: 0 } );
  const refObj = new ml.MeshOrPointsXf( aMop, aXf );
  const fltObj = new ml.MeshOrPointsXf( bMop, bXf );

  const icp = new ml.ICP( fltObj, refObj, 0.12 );
  const props = new ml.ICPProperties();
  props.method = ml.ICPMethod.PointToPoint;
  props.iterLimit = 30;
  assert.equal( props.method.value, ml.ICPMethod.PointToPoint.value, 'enum property round-trips' );
  assert.equal( props.iterLimit, 30, 'scalar property round-trips' );
  icp.setParams( props );

  const xf = icp.calculateTransformation();
  assert.ok( icp.getMeanSqDistToPoint() < 0.05, 'point-to-point ICP converged' );

  xf.delete(); props.delete(); icp.delete();
  refObj.delete(); fltObj.delete();
  aXf.delete(); bXf.delete();
  aMop.delete(); bMop.delete();
  sphere.delete();
}

// MultiwayICP registers several offset copies of one sphere
{
  const sphere = ml.makeUVSphere( 1, 24, 24 );
  const offsets = [ { x: 0, y: 0, z: 0 }, { x: 0.15, y: 0, z: 0 }, { x: 0, y: 0.15, z: 0 } ];
  const mops = [], xfs = [], objs = [];
  for ( const o of offsets ) {
    const mop = ml.MeshOrPoints.fromMesh( sphere );
    const xf = ml.AffineXf3f.translation( o );
    mops.push( mop );
    xfs.push( xf );
    objs.push( new ml.MeshOrPointsXf( mop, xf ) );
  }

  const params = new ml.MultiwayICPSamplingParameters();
  params.samplingVoxelSize = 0.1;
  const mw = new ml.MultiwayICP( objs, params );

  const results = mw.calculateTransformations();  // Float32Array, 12 floats (Matrix3f A + Vector3f b) per object
  assert.equal( results.length, offsets.length * 12, 'a 12-float transform per object' );
  assert.ok( mw.getNumSamples() > 0, 'MultiwayICP formed sample pairs' );
  assert.ok( mw.getMeanSqDistToPoint() < 0.1, 'MultiwayICP converged' );

  mw.delete();
  params.delete();
  for ( const o of objs ) o.delete();
  for ( const x of xfs ) x.delete();
  for ( const m of mops ) m.delete();
  sphere.delete();
}
