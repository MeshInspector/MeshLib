#include "MREFillHoleMetrics.h"
#include "MRMesh/MRTriMath.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBestFit.h"

namespace
{
// This constant modifier was born empirically
constexpr double TriangleAreaModifier = 1e2;
}

namespace MRE
{
using namespace MR;

double ComplexFillMetric::getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const
{
    auto abVec = points[b] - points[a];
    auto bcVec = -abVec;
    auto normA = cross( points[right] - points[b], bcVec );
    auto normC = cross( points[left] - points[a], abVec );

    auto s_Abc_double = normA.length();
    auto s_abC_double = normC.length();
    auto cosAC = dot( normA, normC ) / ( s_Abc_double * s_abC_double );

    if ( cosAC <= -1.0f )
        return DBL_MAX;

    return sqr( sqr( ( 1.0f - cosAC ) / ( 1.0f + cosAC ) ) );
}

double ComplexFillMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const
{
    double aspectRatio = triangleAspectRatio( points[a], points[b], points[c] );
    if ( aspectRatio == DBL_MAX )
        return DBL_MAX;

    double angleABFine = getEdgeMetric( a, b, cOpposit, c );
    if ( angleABFine == DBL_MAX )
        return DBL_MAX;

    double angleBCFine = getEdgeMetric( b, c, aOpposit, a );
    if ( angleBCFine == DBL_MAX )
        return DBL_MAX;

    double normedArea = TriangleAreaModifier * cross( points[b] - points[a], points[c] - points[a] ).length() * reverseCharacteristicTriArea;

    return angleABFine + angleBCFine + aspectRatio + normedArea;
}

ComplexFillMetric::ComplexFillMetric( const Mesh& mesh, EdgeId e0 ) :
    points{ mesh.points }
{
    float maxEdgeLengthSq = 0.0f;
    for ( auto e : leftRing( mesh.topology, e0 ) )
        maxEdgeLengthSq = std::max( maxEdgeLengthSq, mesh.edgeLengthSq( e ) );

    assert( maxEdgeLengthSq > 0.0f );

    if ( maxEdgeLengthSq <= 0.0f )
        reverseCharacteristicTriArea = 1.0f;
    else
        reverseCharacteristicTriArea = 1.0f / maxEdgeLengthSq;

}

ParallelPlaneFillMetric::ParallelPlaneFillMetric( const Mesh& mesh, MR::EdgeId e0, const MR::Plane3f* plane /*= nullptr */ ):
    points{ mesh.points }
{
    if ( plane )
        normal = plane->n.normalized();
    else
    {
        PointAccumulator accum;
        for ( auto e : leftRing( mesh.topology, e0 ) )
            accum.addPoint( mesh.orgPnt( e ) );

        normal = accum.getBestPlanef().n.normalized();
    }
}

double ParallelPlaneFillMetric::getTriangleMetric( const MR::VertId& a, const MR::VertId&, const MR::VertId& c, const MR::VertId&, const MR::VertId& ) const
{
    return std::abs( dot( normal, points[c] - points[a] ) );
}

double ParallelPlaneFillMetric::getEdgeMetric( const MR::VertId& a, const MR::VertId& b, const MR::VertId&, const MR::VertId& ) const
{
    return -std::abs( dot( normal, points[b] - points[a] ) );
}

}
