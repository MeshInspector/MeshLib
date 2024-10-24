#include "MRBoolean.h"
#include "MRFloatGrid.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
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
