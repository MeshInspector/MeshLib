#pragma once
#include "MRVoxelsFwd.h"
#include "MRFloatGrid.h"
// this header includes the whole OpenVDB, so please include it from .cpp files only
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBox.h"

#include "MROpenVDB.h"
#include "MRPch/MRBindingMacros.h"

namespace MR
{

/**
 * \defgroup BasicStructuresGroup Basic Structures
 * \brief This chapter represents documentation about basic structures elements
 * \{
 */

/// this class just hides very complex type of typedef openvdb::FloatGrid
struct MRVOXELS_CLASS OpenVdbFloatGrid : openvdb::FloatGrid
{
    OpenVdbFloatGrid() noexcept = default;
    MR_BIND_IGNORE OpenVdbFloatGrid( openvdb::FloatGrid && in ) : openvdb::FloatGrid( std::move( in ) ) {}
    [[nodiscard]] size_t heapBytes() const { return memUsage(); }
};

MR_BIND_IGNORE inline openvdb::FloatGrid & ovdb( OpenVdbFloatGrid & v ) { return v; }
MR_BIND_IGNORE inline const openvdb::FloatGrid & ovdb( const OpenVdbFloatGrid & v ) { return v; }

/// prohibit unnecessary conversion
MR_BIND_IGNORE inline FloatGrid MakeFloatGrid( const FloatGrid & ) = delete;

/// makes MR::FloatGrid shared pointer taking the contents of the input pointer
MR_BIND_IGNORE inline FloatGrid MakeFloatGrid( openvdb::FloatGrid::Ptr&& p )
{
    if ( !p )
        return {};
    return std::make_shared<OpenVdbFloatGrid>( std::move( *p ) );
}

MR_BIND_IGNORE inline Vector3i fromVdb( const openvdb::Coord & v )
{
    return Vector3i( v.x(), v.y(), v.z() );
}

MR_BIND_IGNORE inline openvdb::Coord toVdb( const Vector3i & v )
{
    return openvdb::Coord( v.x, v.y, v.z );
}

MR_BIND_IGNORE inline openvdb::CoordBBox toVdbBox( const Box3i& box )
{
    return openvdb::CoordBBox( toVdb( box.min ), toVdb( box.max ) );
}

MR_BIND_IGNORE inline openvdb::CoordBBox toVdbBox( const Vector3i& dims )
{
    return openvdb::CoordBBox( toVdb( Vector3i( 0, 0, 0 ) ),
                               toVdb( Vector3i( dims ) - Vector3i::diagonal(1) ) );
}

/// \}

}
