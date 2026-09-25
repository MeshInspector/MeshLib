#pragma once

#include "MRVoxelsFwd.h"
#include "MRPch/MROpenVDB.h"

// The OpenVDB templates used by MeshLib are instantiated once in MRVDBInstantiations.cpp and only referenced from other
// translation units; OpenVDB built with USE_EXPLICIT_INSTANTIATION ships these instantiations itself.
#ifndef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef _WIN32
#   ifdef MRVoxels_EXPORTS
#       define MR_VDB_TEMPLATE_EXPORT __declspec(dllexport)
#       define MR_VDB_TEMPLATE_IMPORT
#   else
#       define MR_VDB_TEMPLATE_EXPORT
#       define MR_VDB_TEMPLATE_IMPORT __declspec(dllimport)
#   endif
#else
#   define MR_VDB_TEMPLATE_EXPORT __attribute__((visibility("default")))
#   define MR_VDB_TEMPLATE_IMPORT
#endif

#ifdef MR_VDB_DEFINE_INSTANTIATIONS
#   define MR_VDB_INSTANTIATE template MR_VDB_TEMPLATE_EXPORT
#   define MR_VDB_INSTANTIATE_CLASS template class MR_VDB_TEMPLATE_EXPORT
#else
#   define MR_VDB_INSTANTIATE extern template MR_VDB_TEMPLATE_IMPORT
#   define MR_VDB_INSTANTIATE_CLASS extern template class MR_VDB_TEMPLATE_IMPORT
#endif

namespace openvdb
{
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME
{
namespace tools
{

MR_VDB_INSTANTIATE FloatGrid::Ptr meshToLevelSet<FloatGrid>( util::NullInterrupter&, const math::Transform&,
    const std::vector<Vec3s>&, const std::vector<Vec3I>&, float );
MR_VDB_INSTANTIATE FloatGrid::Ptr meshToLevelSet<FloatGrid>( util::NullInterrupter&, const math::Transform&,
    const std::vector<Vec3s>&, const std::vector<Vec3I>&, const std::vector<Vec4I>&, float );
MR_VDB_INSTANTIATE FloatGrid::Ptr meshToUnsignedDistanceField<FloatGrid>( util::NullInterrupter&, const math::Transform&,
    const std::vector<Vec3s>&, const std::vector<Vec3I>&, const std::vector<Vec4I>&, float );

MR_VDB_INSTANTIATE void volumeToMesh( const FloatGrid&, std::vector<Vec3s>&, std::vector<Vec3I>&, std::vector<Vec4I>&, double, double, bool );

MR_VDB_INSTANTIATE void csgUnion( FloatGrid&, FloatGrid&, bool, bool );
MR_VDB_INSTANTIATE void csgIntersection( FloatGrid&, FloatGrid&, bool, bool );
MR_VDB_INSTANTIATE void csgDifference( FloatGrid&, FloatGrid&, bool, bool );
MR_VDB_INSTANTIATE FloatGrid::Ptr csgUnionCopy( const FloatGrid&, const FloatGrid& );
MR_VDB_INSTANTIATE FloatGrid::Ptr csgIntersectionCopy( const FloatGrid&, const FloatGrid& );
MR_VDB_INSTANTIATE FloatGrid::Ptr csgDifferenceCopy( const FloatGrid&, const FloatGrid& );

MR_VDB_INSTANTIATE void GridTransformer::transformGrid<QuadraticSampler, FloatGrid>( const FloatGrid&, FloatGrid& ) const;

MR_VDB_INSTANTIATE_CLASS Filter<FloatGrid, FloatGrid, util::NullInterrupter>;

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION
