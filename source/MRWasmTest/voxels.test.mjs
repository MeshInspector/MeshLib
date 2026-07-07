import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rmSync } from 'node:fs';
import { ml, meshToGeometry } from './helpers.mjs';

// mesh -> signed-distance VDB volume -> mesh, plus volume field access
{
  const sphere = ml.makeUVSphere( 1, 24, 24 );
  const params = new ml.MeshToVolumeParams();
  params.voxelSize = { x: 0.1, y: 0.1, z: 0.1 };
  params.type = ml.MeshToVolumeType.Signed;

  const vol = ml.meshToVolume( sphere, params );
  assert.ok( vol.dims.x > 0 && vol.dims.y > 0 && vol.dims.z > 0, 'volume has positive dims' );

  const grid = vol.data;
  const mm = ml.evalGridMinMax( grid );
  assert.ok( mm.min < 0 && mm.max > 0, 'signed distance field straddles zero' );

  const settings = new ml.GridToMeshSettings();
  settings.voxelSize = { x: 0.1, y: 0.1, z: 0.1 };
  settings.isoValue = 0;
  const recon = ml.gridToMesh( grid, settings );
  const g = meshToGeometry( recon, false );
  assert.ok( g.positions.length > 0 && g.indices.length > 0, 'reconstructed mesh is non-empty' );

  recon.delete();
  settings.delete();
  grid.delete();
  vol.delete();
  params.delete();
  sphere.delete();
}

// offsetMesh expands the surface
{
  const sphere = ml.makeUVSphere( 1, 24, 24 );
  const params = new ml.OffsetParameters();
  params.voxelSize = 0.1;
  params.signDetectionMode = ml.SignDetectionMode.OpenVDB;

  const off = ml.offsetMesh( sphere, 0.2, params );
  const g = meshToGeometry( off, false );
  assert.ok( g.positions.length > 0, 'offset mesh is non-empty' );
  let maxR = 0;
  for ( let i = 0; i < g.positions.length; i += 3 )
    maxR = Math.max( maxR, Math.hypot( g.positions[ i ], g.positions[ i + 1 ], g.positions[ i + 2 ] ) );
  assert.ok( maxR > 1.05, `offset pushed the surface outward (maxR=${maxR.toFixed( 3 )})` );

  off.delete();
  params.delete();
  sphere.delete();
}

// VDB voxel file I/O round-trip (NODERAWFS)
{
  const sphere = ml.makeUVSphere( 1, 16, 16 );
  const params = new ml.MeshToVolumeParams();
  params.voxelSize = { x: 0.15, y: 0.15, z: 0.15 };
  const vol = ml.meshToVolume( sphere, params );

  const path = join( tmpdir(), `meshlib-vox-${process.pid}.vdb` );
  ml.VoxelsSave.toAnySupportedFormat( vol, path );
  const loaded = ml.VoxelsLoad.fromAnySupportedFormat( path );
  assert.ok( loaded.length >= 1, 'VDB load returned at least one volume' );
  assert.ok( loaded[ 0 ].dims.x > 0, 'loaded volume has positive dims' );

  for ( const v of loaded )
    v.delete();
  rmSync( path );
  vol.delete();
  params.delete();
  sphere.delete();
}
