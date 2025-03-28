#ifndef MESHLIB_NO_VOXELS

#include "MRVoxels/MRBoolean.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRGTest.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <openvdb/version.h>

namespace MR
{

// below test crashed on Fedora 40 and 41 because of incompatibility between VDB and TBB
TEST( MRMesh, VersionVDB )
{
#if defined TBB_VERSION_PATCH
    spdlog::info( "TBB version: {}.{}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR, TBB_VERSION_PATCH );
#else
    spdlog::info( "TBB version: {}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR );
#endif
    spdlog::info( "OpenVDB version: {}", OPENVDB_LIBRARY_VERSION_STRING );
}

TEST( MRMesh, MeshVoxelsConverterSelfIntersections )
{
    auto torus = makeTorusWithSelfIntersections( 2.f, 1.f, 10, 10 );
    MeshVoxelsConverter converter;
    converter.voxelSize = 0.1f;
    auto grid = converter( torus );
    torus = converter( grid );
    ASSERT_GT( torus.volume(), 0.f );
}

} //namespace MR

#endif //!MESHLIB_NO_VOXELS
