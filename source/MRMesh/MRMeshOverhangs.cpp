#include "MRMeshOverhangs.h"

#include "MRBitSetParallelFor.h"
#include "MRComputeBoundingBox.h"
#include "MRExpandShrink.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MREdgeMetric.h"
#include "MREdgePathsBuilder.h"

#include <cassert>

namespace MR
{

float regionWidth( const MeshPart& mp, const Vector3f& axis, const std::vector<EdgeLoop>& allBoundaries, const std::vector<int>& thisBdIndices )
{
    MR_TIMER
    auto metric = [&] ( EdgeId e0 )->float
    {
        bool incident = false;
        for ( auto e : orgRing( mp.mesh.topology, e0.sym() ) )
        {
            auto f = mp.mesh.topology.left( e );
            if ( f && mp.region->test( f ) )
            {
                incident = true;
                break;
            }
        }
        if ( !incident )
            return FLT_MAX;
        auto ev = mp.mesh.edgeVector( e0 );
        return std::sqrt( ev.lengthSq() - sqr( dot( ev, axis ) ) );
    };


    EdgePathsBuilder b( mp.mesh.topology, metric );
    for ( auto bdId : thisBdIndices )
        for ( auto e : allBoundaries[bdId] )
            b.addStart( mp.mesh.topology.org( e ), 0.0f );

    float maxWidth = b.doneDistance();
    if ( maxWidth == FLT_MAX )
        maxWidth = 0.0f;
    while ( !b.done() )
    {
        b.growOneEdge();
        auto width = b.doneDistance();
        if ( width < FLT_MAX )
            maxWidth = width;
    }
    if ( maxWidth > 0.0f )
        return 2 * maxWidth;

    for ( auto bdId : thisBdIndices )
        for ( auto e : allBoundaries[bdId] )
            for ( auto oe : orgRing( mp.mesh.topology, e ) )
            {
                auto width = metric( oe );
                if ( width < FLT_MAX && width > maxWidth )
                    maxWidth = width;
            }
    return maxWidth;
}

Expected<std::vector<FaceBitSet>> findOverhangs( const Mesh& mesh, const FindOverhangsSettings& settings )
{
    MR_TIMER

    assert( std::abs( settings.axis.lengthSq() - 1.f ) < 1e-6f );
    assert( settings.layerHeight > 0.f );
    assert( settings.maxOverhangDistance > 0.f );
    assert( settings.hops >= 0 );
    const auto minCos = -settings.maxOverhangDistance / std::hypot( settings.layerHeight, settings.maxOverhangDistance );

    const auto xf = settings.xf ? *settings.xf : AffineXf3f();
    const auto isOverhanging = [&] ( FaceId f ) -> bool
    {
        const auto normal = mesh.normal( f );
        const auto cos = dot( settings.axis, xf.A * normal );
        return cos < minCos;
    };

    if ( !reportProgress( settings.progressCb, 0.0f ) )
        return unexpectedOperationCanceled();

    // find faces that might create overhangs
    FaceBitSet faces( mesh.topology.lastValidFace() + 1, false );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&] ( FaceId f )
    {
        faces[f] = isOverhanging( f );
    } );

    if ( !reportProgress( settings.progressCb, 0.2f ) )
        return unexpectedOperationCanceled();

    // smooth out the regions...
    if ( settings.hops > 0 )
    {
        auto smoothFaces = faces;
        expand( mesh.topology, smoothFaces, settings.hops );
        shrink( mesh.topology, smoothFaces, settings.hops );
        faces |= smoothFaces;
    }

    // compute transform from the given axis
    const auto axisXf = xf * AffineXf3f::xfAround( Matrix3f::rotation( Vector3f::plusZ(), settings.axis ), mesh.computeBoundingBox( settings.xf ).center() );
    const auto axisMeshBox = computeBoundingBox( mesh.points, nullptr, &axisXf );

    // filter out face regions with too small overhang distance
    auto regions = MeshComponents::getAllComponents( { mesh, &faces }, MeshComponents::PerVertex );
    if ( !reportProgress( settings.progressCb, 0.3f ) )
        return unexpectedOperationCanceled();

    auto allBds = findRightBoundary( mesh.topology, faces );
    if ( !reportProgress( settings.progressCb, 0.4f ) )
        return unexpectedOperationCanceled();

    std::vector<Box3f> axisBoxes( regions.size() );
    auto keepGoing = ParallelFor( regions, [&] ( size_t i )
    {
        axisBoxes[i] = mesh.computeBoundingBox( &regions[i], &axisXf );
    }, subprogress( settings.progressCb, 0.4f, 0.5f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    // filter out basement faces
    if ( settings.maxBasementAngle >= 0.f )
    {
        const auto maxBasementCos = -std::cos( settings.maxBasementAngle );
        const auto regionCount = regions.size();
        for ( auto i = 0; i < regionCount; ++i )
        {
            auto& region = regions[i];
            auto& axisBox = axisBoxes[i];
            // TODO: basement layer error
            if ( axisBox.min.z == axisMeshBox.min.z )
            {
                auto basementOverhangs = region;
                BitSetParallelFor( region, [&] ( FaceId f )
                {
                    const auto normal = mesh.normal( f );
                    const auto cos = dot( settings.axis, xf.A * normal );
                    if ( cos < maxBasementCos )
                        basementOverhangs.reset( f );
                } );

                region.clear();
                if ( basementOverhangs.none() )
                    continue;

                auto basementRegions = MeshComponents::getAllComponents( { mesh, &basementOverhangs }, MeshComponents::PerVertex );
                if ( basementRegions.size() == 1 )
                {
                    axisBox = mesh.computeBoundingBox( &basementOverhangs, &xf );
                    std::swap( region, basementOverhangs );
                }
                else for ( auto& subregion : basementRegions )
                {
                    axisBoxes.emplace_back( mesh.computeBoundingBox( &subregion, &xf ) );
                    regions.emplace_back( std::move( subregion ) );
                }
            }
        }
    }
    if ( !reportProgress( settings.progressCb, 0.6f ) )
        return unexpectedOperationCanceled();

    keepGoing = ParallelFor( regions, [&] ( size_t i )
    {
        auto& region = regions[i];
        if ( region.none() )
            return;

        auto& axisBox = axisBoxes[i];
        if ( axisBox.size().z > settings.layerHeight )
            return;

        std::vector<int> thisBds;
        for ( int bdI = 0; bdI < allBds.size(); ++bdI )
            if ( region.test( mesh.topology.right( allBds[bdI].front() ) ) )
                thisBds.push_back( bdI );
        auto width = regionWidth( { mesh, &region }, settings.axis, allBds, thisBds );
        if ( width < settings.maxOverhangDistance )
            region.clear();
    }, subprogress( settings.progressCb, 0.6f, 0.95f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    std::erase_if( regions, [] ( const auto& r ) { return r.none(); } );
    if ( !reportProgress( settings.progressCb, 1.0f ) )
        return unexpectedOperationCanceled();

    return regions;
}

} // namespace MR