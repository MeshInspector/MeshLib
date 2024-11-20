#include "MRBoolean.h"

#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRTorus.h"

#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <openvdb/version.h>
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

TEST( MRVoxels, About )
{
#if defined TBB_VERSION_PATCH
    spdlog::info( "TBB version: {}.{}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR, TBB_VERSION_PATCH );
#else
    spdlog::info( "TBB version: {}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR );
#endif
    spdlog::info( "OpenVDB version: {}", OPENVDB_LIBRARY_VERSION_STRING );
}

TEST( MRVoxels, MeshVoxelsConverterSelfIntersections )
{
    auto torus = makeTorusWithSelfIntersections( 2.f, 1.f, 10, 10 );
    MeshVoxelsConverter converter;
    converter.voxelSize = 0.1f;
    auto grid = converter( torus );
    torus = converter( grid );
    ASSERT_GT( torus.volume(), 0.f );
}

} //namespace MR
