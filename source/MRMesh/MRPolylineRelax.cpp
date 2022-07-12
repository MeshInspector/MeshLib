#include "MRBitSetParallelFor.h"
#include "MRPolyline.h"
#include "MRPolylineRelax.h"
#include "MRTimer.h"
#include "MRVector2.h"
#include "MRWriter.h"

namespace
{

template<typename, typename>
struct SubstType
{
};

template<typename A, template<typename> typename C, typename B>
struct SubstType<A, C<B>>
{
    using type = C<A>;
};

}

namespace MR
{

template<typename V>
bool relaxImpl( Polyline<V> &polyline, const RelaxParams &params, ProgressCallback cb )
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
            if ( polyline.topology.isLoneEdge( e0 ) )
                return;
            auto e1 = polyline.topology.next( e0 );

            using VectorD = typename SubstType<double, V>::type;
            VectorD sum;
            sum += VectorD( polyline.destPnt( e0 ) );
            sum += VectorD( polyline.destPnt( e1 ) );

            auto& np = newPoints[v];
            auto pushForce = params.force * ( V( sum / 2. ) - np );
            np += pushForce;
        }, internalCb );
        polyline.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

template<typename V>
bool relaxKeepAreaImpl( Polyline<V> &polyline, const RelaxParams &params, ProgressCallback cb )
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
            if ( polyline.topology.isLoneEdge( e0 ) )
                return;
            auto e1 = polyline.topology.next( e0 );

            using VectorD = typename SubstType<double, V>::type;
            VectorD sum;
            sum += VectorD( polyline.destPnt( e0 ) );
            sum += VectorD( polyline.destPnt( e1 ) );

            vertPushForces[v] = params.force * ( V( sum / 2. ) - polyline.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;

        newPoints = polyline.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = polyline.topology.edgeWithOrg( v );
            if ( polyline.topology.isLoneEdge( e0 ) )
                return;
            auto e1 = polyline.topology.next( e0 );

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

bool relax( Polyline2 &polyline, const RelaxParams &params, ProgressCallback cb )
{
    return relaxImpl( polyline, params, std::move(cb) );
}

bool relax( Polyline3 &polyline, const RelaxParams &params, ProgressCallback cb )
{
    return relaxImpl( polyline, params, std::move(cb) );
}

bool relaxKeepArea( Polyline2 &polyline, const RelaxParams &params, ProgressCallback cb )
{
    return relaxKeepAreaImpl( polyline, params, std::move(cb) );
}

bool relaxKeepArea( Polyline3 &polyline, const RelaxParams &params, ProgressCallback cb )
{
    return relaxKeepAreaImpl( polyline, params, std::move(cb) );
}

} // namespace MR