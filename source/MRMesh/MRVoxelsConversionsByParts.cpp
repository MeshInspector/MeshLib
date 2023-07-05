#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRVoxelsConversionsByParts.h"
#include "MREdgePaths.h"
#include "MRMesh.h"
#include "MRMeshTrimWithPlane.h"
#include "MRPlane3.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRVDBConversions.h"

#include "MRGTest.h"
#include "MRFloatGrid.h"

#include <fmt/format.h>

namespace
{

using namespace MR;

void clearPartMapping( PartMapping& mapping )
{
    if ( mapping.src2tgtFaces )
        mapping.src2tgtFaces->clear();
    if ( mapping.src2tgtVerts )
        mapping.src2tgtVerts->clear();
    if ( mapping.src2tgtEdges )
        mapping.src2tgtEdges->clear();
    if ( mapping.tgt2srcFaces )
        mapping.tgt2srcFaces->clear();
    if ( mapping.tgt2srcVerts )
        mapping.tgt2srcVerts->clear();
    if ( mapping.tgt2srcEdges )
        mapping.tgt2srcEdges->clear();
}

void sortEdgePaths( const Mesh& mesh, std::vector<EdgePath>& paths )
{
    std::sort( paths.begin(), paths.end(), [&] ( const EdgePath& ep1, const EdgePath& ep2 )
    {
        auto org1 = mesh.orgPnt( ep1.front() );
        auto org2 = mesh.orgPnt( ep2.front() );
        auto dest1 = mesh.destPnt( ep1.front() );
        auto dest2 = mesh.destPnt( ep2.front() );
        if ( org1 != org2 )
            return std::tie( org1.y, org1.z ) < std::tie( org2.y, org2.z );
        else
            return std::tie( dest1.y, dest1.z ) < std::tie( dest2.y, dest2.z );
    } );
}

} // namespace

namespace MR
{

VoidOrErrStr
mergeGridPart( Mesh &mesh, std::vector<EdgePath> &cutContours, const FunctionVolume& volume, const Vector3i& offset,
               float leftCutPosition, float rightCutPosition, const MergeGridPartSettings &settings )
{
    MR_TIMER

    auto res = functionVolumeToMesh( volume, {
        .iso = 0.f,
        .lessInside = true,
        .omitNaNCheck = true,
    } );
    if ( !res.has_value() )
        return unexpected( res.error() );
    auto part = std::move( *res );

    part.transform( AffineXf3f::translation( mult( Vector3f( offset ), volume.voxelSize ) ) );

    if ( settings.preCut )
        settings.preCut( part, leftCutPosition, rightCutPosition );

    std::vector<EdgePath> leftCutContours;
    if ( leftCutPosition != -FLT_MAX )
    {
        trimWithPlane( part, Plane3f { Vector3f::plusX(), leftCutPosition }, &leftCutContours );
        sortEdgePaths( part, leftCutContours );
    }

    std::vector<EdgePath> rightCutContours;
    if ( rightCutPosition != +FLT_MAX )
    {
        trimWithPlane( part, -Plane3f {Vector3f::plusX(), rightCutPosition }, &rightCutContours );
        reverse( rightCutContours );
        sortEdgePaths( part, rightCutContours );
    }

    if ( settings.postCut )
        settings.postCut( part );

    auto mapping = settings.mapping;
    clearPartMapping( mapping );

    if ( leftCutContours.empty() && cutContours.empty() )
    {
        WholeEdgeHashMap src2tgtEdges;
        if ( !mapping.src2tgtEdges )
            mapping.src2tgtEdges = &src2tgtEdges;

        mesh.addPartByMask( part, part.topology.getValidFaces(), mapping );

        if ( settings.postMerge )
            settings.postMerge( mesh, mapping );

        for ( auto& contour : rightCutContours )
        {
            for ( auto& e : contour )
            {
                const auto ue = ( *mapping.src2tgtEdges )[e];
                e = e.even() ? ue : ue.sym();
            }
        }
        cutContours = std::move( rightCutContours );

        return {};
    }

    if ( cutContours.size() != leftCutContours.size() )
        return unexpected( "Mesh cut contours mismatch" );
    for ( auto i = 0u; i < cutContours.size(); ++i )
        if ( cutContours[i].size() != leftCutContours[i].size() )
            return unexpected( "Mesh cut contours mismatch" );

    WholeEdgeHashMap src2tgtEdges;
    if ( !mapping.src2tgtEdges )
        mapping.src2tgtEdges = &src2tgtEdges;

    mesh.addPartByMask( part, part.topology.getValidFaces(), false, cutContours, leftCutContours, mapping );

    if ( settings.postMerge )
        settings.postMerge( mesh, mapping );

    for ( auto& contour : rightCutContours )
    {
        for ( auto& e : contour )
        {
            const auto ue = ( *mapping.src2tgtEdges )[e];
            e = e.even() ? ue : ue.sym();
        }
    }
    cutContours = std::move( rightCutContours );

    return {};
}

Expected<Mesh, std::string>
gridToMeshByParts( const VoxelValueGetter& getter, const Vector3i &dimensions, const Vector3f &voxelSize,
                   const GridToMeshByPartsSettings &settings, const MergeGridPartSettings &mergeSettings )
{
    MR_TIMER

    constexpr float cMemOverhead = 1.25f;
    const auto maxSliceMemoryUsage = size_t( float( dimensions.y * dimensions.z * sizeof( float ) ) * cMemOverhead );
    const auto maxSliceCount = settings.maxGridPartMemoryUsage / maxSliceMemoryUsage;
    if ( maxSliceCount < settings.stripeOverlap + 1 )
    {
        return unexpected( fmt::format( "The specified grid memory usage limit is too low: at least {} required",
                                                 bytesString( ( settings.stripeOverlap + 1 ) * maxSliceMemoryUsage ) ) );
    }

    const size_t width = dimensions.x - settings.stripeOverlap;
    size_t stripeSize_ = maxSliceCount - settings.stripeOverlap;
    size_t stripeCount_ = width / stripeSize_;
    size_t lastStripeSize_ = width - stripeSize_ * stripeCount_;
    while ( lastStripeSize_ != 0 && lastStripeSize_ < settings.stripeOverlap - 1 )
    {
        stripeSize_ -= 1;
        stripeCount_ = width / stripeSize_;
        lastStripeSize_ = width - stripeSize_ * stripeCount_;
    }

    const size_t stripeSize = stripeSize_ + settings.stripeOverlap;
    const size_t stripeCount = stripeCount_ + size_t( lastStripeSize_ != 0 );
    assert( ( stripeSize - settings.stripeOverlap ) * stripeCount >= width );

    Mesh result;
    std::vector<EdgePath> cutContours;
    for ( auto stripe = 0; stripe < stripeCount; stripe++ )
    {
        const auto begin = stripe * ( stripeSize - settings.stripeOverlap );
        const auto end = std::min( begin + stripeSize, (size_t)dimensions.x );

        auto leftCutPosition = ( (float)begin + (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( begin == 0 )
            leftCutPosition = -FLT_MAX;
        auto rightCutPosition = ( (float)end - (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( end == dimensions.x )
            rightCutPosition = +FLT_MAX;

        const Vector3i offset( begin, 0, 0 );
        VoxelValueGetter offsetGetter = [&] ( const Vector3i& pos )
        {
            return getter( pos + offset );
        };

        FunctionVolume volume {
            .data = offsetGetter,
            .dims = { int( end - begin ), dimensions.y, dimensions.z },
            .voxelSize = voxelSize,
            .min = -FLT_MAX,
            .max = +FLT_MAX,
        };
        const auto res = mergeGridPart( result, cutContours, volume, offset, leftCutPosition, rightCutPosition, mergeSettings );
        if ( !res.has_value() )
            return unexpected( res.error() );
    }
    return std::move( result );
}

# if false
TEST( MRMesh, gridToMeshByParts )
{
    const Vector3i dimensions { 101, 101, 101 };
    constexpr float radius = 50.f;
    constexpr Vector3f center { 50.f, 50.f, 50.f };

    GridPartBuilder builder = [&] ( size_t begin, size_t end )
    {
        auto grid = MakeFloatGrid( openvdb::FloatGrid::create() );
        grid->setGridClass( openvdb::GRID_LEVEL_SET );

        auto accessor = grid->getAccessor();
        for ( int x = (int)begin; x < end; ++x )
        {
            for ( int y = 0; y < dimensions.y; ++y )
            {
                for ( int z = 0; z < dimensions.z; ++z )
                {
                    const Vector3f pos( (float)x, (float)y, (float)z );
                    const auto dist = ( center - pos ).length();
                    accessor.setValue( { x, y, z }, dist - radius );
                }
            }
        }

        return grid;
    };

    constexpr float voxelSize = 0.01f;
    auto mesh = gridToMeshByParts( builder, dimensions, Vector3f::diagonal( voxelSize ), {
        .maxGridPartMemoryUsage = 2 * ( 1 << 20 ), // 2 MiB
    } );
    EXPECT_TRUE( mesh.has_value() );

    if ( mesh.has_value() )
    {
        constexpr auto r = radius * voxelSize;
        constexpr auto expectedVolume = 4.f * PI_F * r * r * r / 3.f;
        const auto actualVolume = mesh->volume();
        EXPECT_NEAR( expectedVolume, actualVolume, 0.001f );
    }
}
# endif

} // namespace MR
#endif
