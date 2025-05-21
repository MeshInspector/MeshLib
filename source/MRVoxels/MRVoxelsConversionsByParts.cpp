#include "MRVoxelsConversionsByParts.h"
#include "MRVoxelsVolume.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"
#include "MRMarchingCubes.h"

#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRTriMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTrimWithPlane.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRMapEdge.h"

#include "MRPch/MRFmt.h"

namespace
{

using namespace MR;

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

template <typename Volume>
Expected<void>
mergeVolumePart( Mesh &mesh, std::vector<EdgePath> &cutContours, Volume &&volume,
               float leftCutPosition, float rightCutPosition, const MergeVolumePartSettings &settings )
{
    MR_TIMER;

    Expected<Mesh> res;
    if constexpr ( std::is_same_v<Volume, VdbVolume> )
    {
        res = gridToMesh( std::move( volume.data ), GridToMeshSettings {
            .voxelSize = volume.voxelSize,
            .isoValue = 0.f,
        } );
    }
    else if constexpr ( std::is_same_v<Volume, SimpleVolumeMinMax> )
    {
        res = marchingCubes( volume, {
            .iso = 0.f,
            .lessInside = true,
            .freeVolume = [&volume]
            {
                Timer t( "~SimpleVolumeMinMax" );
                volume = {};
            }
        } );
    }
    else if constexpr ( std::is_same_v<Volume, FunctionVolume> )
    {
        res = marchingCubes( volume, {
            .iso = 0.f,
            .lessInside = true
        } );
    }
    else
    {
        static_assert( !sizeof( Volume ), "Unsupported voxel volume type." );
    }
    if ( !res.has_value() )
        return unexpected( res.error() );
    auto part = std::move( *res );

    if ( settings.origin != Vector3f() )
        part.transform( AffineXf3f::translation( settings.origin ) );

    if ( settings.preCut )
        settings.preCut( part, leftCutPosition, rightCutPosition );

    std::vector<EdgePath> leftCutContours;
    if ( leftCutPosition != -FLT_MAX )
    {
        trimWithPlane( part, { .plane = Plane3f { Vector3f::plusX(), leftCutPosition } }, { .outCutContours = &leftCutContours } );
        sortEdgePaths( part, leftCutContours );
    }

    std::vector<EdgePath> rightCutContours;
    if ( rightCutPosition != +FLT_MAX )
    {
        trimWithPlane( part, { .plane = -Plane3f{Vector3f::plusX(), rightCutPosition } }, { .outCutContours = &rightCutContours } );
        reverse( rightCutContours );
        sortEdgePaths( part, rightCutContours );
    }

    if ( settings.postCut )
        settings.postCut( part );

    auto mapping = settings.mapping;
    mapping.clear();

    if ( leftCutContours.empty() && cutContours.empty() )
    {
        WholeEdgeMapOrHashMap src2tgtEdges;
        if ( !mapping.src2tgtEdges )
            mapping.src2tgtEdges = &src2tgtEdges;

        mesh.addMeshPart( part, mapping );

        if ( settings.postMerge )
            settings.postMerge( mesh, mapping );

        for ( auto& contour : rightCutContours )
        {
            for ( auto& e : contour )
                e = mapEdge( src2tgtEdges, e );
        }
        cutContours = std::move( rightCutContours );

        return {};
    }

    if ( cutContours.size() != leftCutContours.size() )
        return unexpected( "Mesh cut contours mismatch" );
    for ( auto i = 0u; i < cutContours.size(); ++i )
        if ( cutContours[i].size() != leftCutContours[i].size() )
            return unexpected( "Mesh cut contours mismatch" );

    WholeEdgeMapOrHashMap src2tgtEdges;
    if ( !mapping.src2tgtEdges )
        mapping.src2tgtEdges = &src2tgtEdges;

    mesh.addMeshPart( part, false, cutContours, leftCutContours, mapping );

    if ( settings.postMerge )
        settings.postMerge( mesh, mapping );

    for ( auto& contour : rightCutContours )
    {
        for ( auto& e : contour )
            e = mapEdge( src2tgtEdges, e );
    }
    cutContours = std::move( rightCutContours );

    return {};
}

template <typename Volume>
Expected<Mesh>
volumeToMeshByParts( const VolumePartBuilder<Volume> &builder, const Vector3i &dimensions, const Vector3f &voxelSize,
                     const VolumeToMeshByPartsSettings &settings, const MergeVolumePartSettings &mergeSettings )
{
    MR_TIMER;

    constexpr float cMemOverhead = 1.25f;
    const auto maxSliceMemoryUsage = size_t( float( dimensions.y * dimensions.z * sizeof( float ) ) * cMemOverhead );
    const auto maxSliceCount = settings.maxVolumePartMemoryUsage / maxSliceMemoryUsage;
    if ( maxSliceCount < settings.stripeOverlap + 1 )
    {
        return unexpected( fmt::format( "The specified volume memory usage limit is too low: at least {} required",
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

        std::optional<Vector3i> offset;
        auto volume = builder( (int)begin, (int)end, offset );
        if ( !volume.has_value() )
            return unexpected( volume.error() );

        auto mergeOffsetSettings = mergeSettings;
        if ( offset )
            mergeOffsetSettings.origin = mult( Vector3f( *offset ), voxelSize );

        auto leftCutPosition = ( (float)begin + (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( begin == 0 )
            leftCutPosition = -FLT_MAX;
        auto rightCutPosition = ( (float)end - (float)settings.stripeOverlap / 2.f ) * voxelSize.x;
        if ( end == dimensions.x )
            rightCutPosition = +FLT_MAX;

        const auto res = mergeVolumePart( result, cutContours, std::move( *volume ), leftCutPosition, rightCutPosition, mergeOffsetSettings );
        if ( !res.has_value() )
            return unexpected( res.error() );
    }
    return result;
}

template MRVOXELS_API Expected<void> mergeVolumePart<SimpleVolumeMinMax>( Mesh&, std::vector<EdgePath>&, SimpleVolumeMinMax&&, float, float, const MergeVolumePartSettings& );
template MRVOXELS_API Expected<void> mergeVolumePart<VdbVolume>( Mesh&, std::vector<EdgePath>&, VdbVolume&&, float, float, const MergeVolumePartSettings& );
template MRVOXELS_API Expected<void> mergeVolumePart<FunctionVolume>( Mesh&, std::vector<EdgePath>&, FunctionVolume&&, float, float, const MergeVolumePartSettings& );

template MRVOXELS_API Expected<Mesh> volumeToMeshByParts<SimpleVolumeMinMax>( const VolumePartBuilder<SimpleVolumeMinMax>&, const Vector3i&, const Vector3f&, const VolumeToMeshByPartsSettings&, const MergeVolumePartSettings& );
template MRVOXELS_API Expected<Mesh> volumeToMeshByParts<VdbVolume>( const VolumePartBuilder<VdbVolume>&, const Vector3i&, const Vector3f&, const VolumeToMeshByPartsSettings&, const MergeVolumePartSettings& );
template MRVOXELS_API Expected<Mesh> volumeToMeshByParts<FunctionVolume>( const VolumePartBuilder<FunctionVolume>&, const Vector3i&, const Vector3f&, const VolumeToMeshByPartsSettings&, const MergeVolumePartSettings& );

} // namespace MR
