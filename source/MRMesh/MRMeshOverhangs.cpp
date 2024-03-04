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

    // compute transform from the given axis
    const auto xf = settings.xf ? *settings.xf : AffineXf3f();
    const auto axisXf = xf * AffineXf3f::xfAround( Matrix3f::rotation( Vector3f::plusZ(), settings.axis ), mesh.computeBoundingBox( settings.xf ).center() );
    const auto axisMeshBox = computeBoundingBox( mesh.points, nullptr, &axisXf );

    // find the lowest layer's faces (never considered as an overhang)
    const auto basementTop = axisMeshBox.min.z + settings.layerHeight;
    VertBitSet basementVerts( mesh.topology.lastValidVert() + 1, false );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        basementVerts[v] = ( axisXf( mesh.points[v] ).z <= basementTop );
    } );
    const auto basementFaces = getInnerFaces( mesh.topology, basementVerts );

    const auto isOverhanging = [&] ( FaceId f ) -> bool
    {
        if ( basementFaces.test( f ) )
            return false;
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

    // filter out face regions with too small overhang distance
    auto regions = MeshComponents::getAllComponents( { mesh, &faces }, MeshComponents::PerVertex );

    if ( !reportProgress( settings.progressCb, 0.3f ) )
        return unexpectedOperationCanceled();

    auto allBds = findRightBoundary( mesh.topology, faces );
    if ( !reportProgress( settings.progressCb, 0.4f ) )
        return unexpectedOperationCanceled();
    auto keepGoing = ParallelFor( regions, [&] ( size_t i )
    {
        auto& region = regions[i];
        const auto axisBox = mesh.computeBoundingBox( &region, &axisXf );
        if ( axisBox.size().z > settings.layerHeight )
            return;

        std::vector<int> thisBds;
        for ( int bdI = 0; bdI < allBds.size(); ++bdI )
            if ( region.test( mesh.topology.right( allBds[bdI].front() ) ) )
                thisBds.push_back( bdI );
        auto width = regionWidth( { mesh,&region }, settings.axis, allBds, thisBds );
        if ( width < settings.maxOverhangDistance )
            region.clear();
    }, subprogress( settings.progressCb, 0.4f, 0.95f ) );
    
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    regions.erase( std::remove_if( regions.begin(), regions.end(), [] ( const auto& r )
    {
        return r.empty();
    } ), regions.end() );

    if ( !reportProgress( settings.progressCb, 1.0f ) )
        return unexpectedOperationCanceled();

    return regions;
}

} // namespace MR