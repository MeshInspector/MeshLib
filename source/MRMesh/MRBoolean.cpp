#include "MRBoolean.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRFloatGrid.h"
#include "MRTimer.h"
#include "MRPch/MROpenvdb.h"
#include "MRPch/MRSpdlog.h"
#include <filesystem>

namespace MR
{

FloatGrid MeshVoxelsConverter::operator() ( const ObjectMesh & obj ) const
{ 
    return meshToLevelSet( *obj.mesh(), obj.xf(), Vector3f::diagonal( voxelSize ), surfaceOffset, callBack );
}

Mesh MeshVoxelsConverter::operator() ( const FloatGrid & grid ) const
{ 
    auto res = gridToMesh( grid, GridToMeshSettings{
        .voxelSize = Vector3f::diagonal( voxelSize ),
        .isoValue = offsetVoxels,
        .adaptivity = adaptivity,
        .cb = callBack
    } );
    if ( !res.has_value() )
    {
        spdlog::error( res.error() );
        return {};
    }
    return *res;
}

} //namespace MR
#endif