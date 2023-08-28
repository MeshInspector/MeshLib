#pragma once

#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVDBConversions.h"
#include "MRAffineXf3.h"
#include "MRFloatGrid.h"

namespace MR
{

// converter of meshes in or from signed distance volumetric representation
struct MeshVoxelsConverter
{
    // both in and from
    float voxelSize = 0.001f;

    // to voxels:
    float surfaceOffset = 3; // number voxels around surface to calculate distance in (should be positive)

    ProgressCallback callBack;

    FloatGrid operator() ( const MeshPart & mp, const AffineXf3f& xf = {} ) const
        { return meshToLevelSet( mp, xf, Vector3f::diagonal( voxelSize ), surfaceOffset, callBack); }
    MRMESH_API FloatGrid operator() ( const ObjectMesh & obj ) const;

    // from voxels:
    float offsetVoxels = 0;   // the value is in voxels (not in meters!), 0 for no-offset
    float adaptivity = 0;     // [0, 1] ratio of combining small triangles into bigger ones

    MRMESH_API Mesh operator() ( const FloatGrid & grid ) const;
};

} //namespace MR

#endif