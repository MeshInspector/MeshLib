#include "MRBitSetParallelFor.h"
#include "MRPolyline.h"
#include "MRPolylineRelax.h"
#include "MRTimer.h"
#include "MRWriter.h"

namespace MR
{

bool relax( Polyline3 &polyline, const PolylineRelaxParams &params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    MR_WRITER(polyline)

    VertCoords newPoints;
    const auto& zone = polyline.topology.getValidVerts();

    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        ProgressCallback internalCb;
        if ( cb )
        {
            internalCb = [&]( float p )
            {
                return cb(( float( i ) + p ) / float( params.iterations ));
            };
        }

        newPoints = polyline.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = polyline.topology.edgeWithOrg( v );
            auto e1 = polyline.topology.next( e0 );
            if ( !e0.valid() || !e1.valid() )
                return;

            Vector3d sum;
            sum += Vector3d( polyline.points[polyline.topology.dest( e0 )] );
            sum += Vector3d( polyline.points[polyline.topology.dest( e1 )] );

            auto& np = newPoints[v];
            auto pushForce = params.force * ( Vector3f{sum / 2.} - np );
            np += pushForce;
        }, internalCb );
        polyline.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

bool relaxKeepArea( Polyline3 &polyline, const PolylineRelaxParams &params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    MR_WRITER(polyline)

    VertCoords newPoints;
    const auto& zone = polyline.topology.getValidVerts();
    std::vector<Vector3f> vertPushForces( zone.size() );

    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        ProgressCallback internalCb1, internalCb2;
        if ( cb )
        {
            internalCb1 = [&] ( float p )
            {
                return cb( ( float( i ) + p * 0.5f ) / float( params.iterations ) );
            };
            internalCb2 = [&] ( float p )
            {
                return cb( ( float( i ) + p * 0.5f + 0.5f ) / float( params.iterations ) );
            };
        }

        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = polyline.topology.edgeWithOrg( v );
            auto e1 = polyline.topology.next( e0 );
            if ( !e0.valid() || !e1.valid() )
                return;

            Vector3d sum;
            sum += Vector3d( polyline.points[polyline.topology.dest( e0 )] );
            sum += Vector3d( polyline.points[polyline.topology.dest( e1 )] );

            vertPushForces[v] = params.force * ( Vector3f{sum / 2.} - polyline.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;

        newPoints = polyline.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = polyline.topology.edgeWithOrg( v );
            auto e1 = polyline.topology.next( e0 );
            if ( !e0.valid() || !e1.valid() )
                return;

            auto& np = newPoints[v];
            np += vertPushForces[v];
            auto modifier = 1.0f / 2.0f;
            np -= ( vertPushForces[polyline.topology.dest( e0 )] * modifier );
            np -= ( vertPushForces[polyline.topology.dest( e1 )] * modifier );
        }, internalCb2 );
        polyline.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }

    return keepGoing;
}

} // namespace MR