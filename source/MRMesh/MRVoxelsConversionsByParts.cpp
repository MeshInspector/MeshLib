#include "MRVoxelsConversionsByParts.h"
#include "MREdgePaths.h"
#include "MRMesh.h"
#include "MRMeshTrimWithPlane.h"
#include "MRPlane3.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRVDBConversions.h"

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

tl::expected<void, std::string>
mergeGridPart( Mesh &mesh, std::vector<EdgePath> &cutContours, FloatGrid &&grid, const Vector3f &voxelSize,
               float leftCutPosition, float rightCutPosition, const MergeGridPartSettings &settings )
{
    MR_TIMER

    Timer timer( "convert grid to mesh" );
    auto res = gridToMesh( std::move( grid ), GridToMeshSettings {
            .voxelSize = voxelSize,
    } );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );
    auto part = std::move( *res );

    if ( settings.preCut )
    {
        timer.restart( "pre-cut callback" );
        settings.preCut( part, leftCutPosition, rightCutPosition );
    }

    timer.restart( "cut part" );
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
    {
        timer.restart( "post-cut callback" );
        settings.postCut( part );
    }

    auto mapping = settings.mapping;
    clearPartMapping( mapping );

    if ( mesh.points.empty() )
    {
        mesh = std::move( part );
        cutContours = std::move( rightCutContours );

        if ( settings.postMerge )
        {
            timer.restart( "post-merge callback" );
            settings.postMerge( mesh, mapping );
        }

        return {};
    }

    if ( cutContours.size() != leftCutContours.size() )
        return tl::make_unexpected( "Mesh cut contours mismatch" );
    for ( auto i = 0u; i < cutContours.size(); ++i )
        if ( cutContours[i].size() != leftCutContours[i].size() )
            return tl::make_unexpected( "Mesh cut contours mismatch" );

    timer.restart( "merge part" );
    WholeEdgeHashMap src2tgtEdges;
    if ( !mapping.src2tgtEdges )
        mapping.src2tgtEdges = &src2tgtEdges;

    mesh.addPartByMask( part, part.topology.getValidFaces(), false, cutContours, leftCutContours, mapping );

    if ( settings.postMerge )
    {
        timer.restart( "post-merge callback" );
        settings.postMerge( mesh, mapping );
    }

    timer.restart( "convert cut contours" );
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

tl::expected<Mesh, std::string>
gridToMeshByParts( const GridPartBuilder &builder, const Vector3i &dimensions, const Vector3f &voxelSize,
                   const GridToMeshByPartsSettings &settings, const MergeGridPartSettings &mergeSettings )
{
    MR_TIMER

    constexpr float cMemOverhead = 1.25f;
    const auto maxSliceMemoryUsage = size_t( float( dimensions.y * dimensions.z * sizeof( float ) ) * cMemOverhead );
    const auto maxSliceCount = settings.maxGridPartMemoryUsage / maxSliceMemoryUsage;
    if ( maxSliceCount < settings.stripeOverlap + 1 )
    {
        return tl::make_unexpected( fmt::format( "The specified grid memory usage limit is too low: at least {} required",
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

        GridPartBuilder::result_type grid;
        {
            MR_NAMED_TIMER( "build grid" )
            grid = builder( begin, end );
        }
        if ( !grid.has_value() )
            return tl::make_unexpected( grid.error() );

        auto leftCutPosition = ( (float)begin + (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( begin == 0 )
            leftCutPosition = -FLT_MAX;
        auto rightCutPosition = ( (float)end - (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( end == dimensions.x )
            rightCutPosition = +FLT_MAX;

        const auto res = mergeGridPart( result, cutContours, std::move( *grid ), voxelSize, leftCutPosition, rightCutPosition, mergeSettings );
        if ( !res.has_value() )
            return tl::make_unexpected( res.error() );
    }
    return std::move( result );
}

} // namespace MR
