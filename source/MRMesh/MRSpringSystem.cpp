#include "MRSpringSystem.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

float springDiscrepancyAtVertex( const Mesh & mesh, VertId v, const Vector3f & vpos, const UndirectedEdgeMetric & springRestLength )
{
    float res = 0;
    for ( auto e : orgRing( mesh.topology, v ) )
    {
        res += sqr( ( mesh.destPnt( e ) - vpos ).length() - springRestLength( e ) );
    }
    return res;
}

Vector3f springBestVertexPos( const Mesh & mesh, VertId v, const UndirectedEdgeMetric & springRestLength )
{
    int num = 0;
    Vector3f sum;
    const auto vpos0 = mesh.points[v];
    for ( auto e : orgRing( mesh.topology, v ) )
    {
        const auto dpos = mesh.destPnt( e );
        sum += dpos + springRestLength( e ) * ( vpos0 - dpos ).normalized();
        ++num;
    }
    Vector3f d = sum / float( num ) - vpos0;

    const auto disc0 = springDiscrepancyAtVertex( mesh, v, vpos0, springRestLength );
    const int maxIter = 10;
    for ( int i = 0; i < maxIter; ++i )
    {
        auto vpos = vpos0 + d;
        auto disc = springDiscrepancyAtVertex( mesh, v, vpos, springRestLength );
        if ( disc < disc0 )
            return vpos;
        d *= 0.5f;
    }
    return vpos0;
}

void solveSpringSystem( Mesh& mesh, const SpringSystemSettings & settings )
{
    MR_TIMER
    assert( settings.numIters > 0 );
    assert( settings.springRestLength );

    auto nextPoints = mesh.points;
    for ( int i = 0; i < settings.numIters; ++i )
    {
        BitSetParallelFor( mesh.topology.getVertIds( settings.region ), [&]( VertId v )
        {
            nextPoints[v] = 0.5f * ( mesh.points[v] + springBestVertexPos( mesh, v, settings.springRestLength ) );
        } );
        mesh.points = nextPoints;
    }
}

} //namespace MR
