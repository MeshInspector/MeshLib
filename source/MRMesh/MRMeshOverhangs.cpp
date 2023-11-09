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

float regionWidth( const MeshPart& mp, const Vector3f& axis )
{
    auto boundaries = findRightBoundary( mp.mesh.topology, mp.region );
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
    for ( const auto& bd : boundaries )
        for ( auto e : bd )
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

    for ( const auto& bd : boundaries )
        for ( auto e : bd )
            for ( auto oe : orgRing( mp.mesh.topology, e ) )
            {
                auto width = metric( oe );
                if ( width < FLT_MAX && width > maxWidth )
                    maxWidth = width;
            }
    return maxWidth;
}

std::vector<FaceBitSet> findOverhangs( const Mesh& mesh, const FindOverhangsSettings& settings )
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

    // find faces that might create overhangs
    FaceBitSet faces( mesh.topology.lastValidFace() + 1, false );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&] ( FaceId f )
    {
        faces[f] = isOverhanging( f );
    } );

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
    auto regions = MeshComponents::getAllComponents( { mesh, &faces } );
    ParallelFor( regions, [&] ( size_t i )
    {
        auto& region = regions[i];
        const auto axisBox = mesh.computeBoundingBox( &region, &axisXf );
        if ( axisBox.size().z > settings.layerHeight )
            return;
        // don't include the basement region
        if ( axisBox.min.z == axisMeshBox.min.z )
        {
            region.clear();
            return;
        }
        auto width = regionWidth( { mesh,&region }, settings.axis );
        if ( width < settings.maxOverhangDistance )
            region.clear();
    } );
    
    regions.erase( std::remove_if( regions.begin(), regions.end(), [] ( const auto& r )
    {
        return r.empty();
    } ), regions.end() );

    return regions;
}

} // namespace MR