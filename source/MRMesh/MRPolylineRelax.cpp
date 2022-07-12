#include "MRPolylineRelax.h"
#include "MRBitSetParallelFor.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRVector2.h"
#include "MRWriter.h"

namespace MR
{

template<typename V>
bool relax( Polyline<V> &polyline, const RelaxParams &params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    MR_WRITER(polyline)

    Vector<V, VertId> newPoints;
    const auto& zone = polyline.topology.getVertIds( params.region );

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
            assert( !polyline.topology.isLoneEdge( e0 ) );
            auto e1 = polyline.topology.next( e0 );
            if ( e0 == e1 )
                return;

            auto mp = ( polyline.destPnt( e0 ) + polyline.destPnt( e1 ) ) / 2.f;

            auto& np = newPoints[v];
            auto pushForce = params.force * ( mp - np );
            np += pushForce;
        }, internalCb );
        polyline.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

template<typename V>
bool relaxKeepArea( Polyline<V> &polyline, const RelaxParams &params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    MR_WRITER(polyline)

    Vector<V, VertId> newPoints;
    const auto& zone = polyline.topology.getVertIds( params.region );
    std::vector<V> vertPushForces( zone.size() );

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
            assert( !polyline.topology.isLoneEdge( e0 ) );
            auto e1 = polyline.topology.next( e0 );
            if ( e0 == e1 )
                return;

            auto mp = ( polyline.destPnt( e0 ) + polyline.destPnt( e1 ) ) / 2.f;

            vertPushForces[v] = params.force * ( mp - polyline.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;

        newPoints = polyline.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = polyline.topology.edgeWithOrg( v );
            assert( !polyline.topology.isLoneEdge( e0 ) );
            auto e1 = polyline.topology.next( e0 );
            if ( e0 == e1 )
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

template MRMESH_API bool relax<Vector2f>( Polyline2& polyline, const RelaxParams& params, ProgressCallback cb );
template MRMESH_API bool relax<Vector3f>( Polyline3& polyline, const RelaxParams& params, ProgressCallback cb );

template MRMESH_API bool relaxKeepArea<Vector2f>( Polyline2 &polyline, const RelaxParams &params, ProgressCallback cb );
template MRMESH_API bool relaxKeepArea<Vector3f>( Polyline3 &polyline, const RelaxParams &params, ProgressCallback cb );

} // namespace MR