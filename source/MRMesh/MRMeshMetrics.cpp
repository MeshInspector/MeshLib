#include "MRMeshMetrics.h"
#include "MRId.h"
#include "MRMeshDelone.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"

namespace 
{
// Big value, but less then DBL_MAX, to be able to pass some bad triangulations instead of breaking it
constexpr double BadTriangulationMetric = 1e100;
}

namespace MR
{

double CircumscribedFillMetric::getEdgeMetric( const VertId& /*a*/, const VertId& /*b*/, const VertId& /*left*/, const VertId& /*right*/ ) const
{
    return 0.0;
}

double CircumscribedFillMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const
{
    return circumcircleDiameter( points[a], points[b], points[c] );
}

double PlaneFillMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& /*aOpposit*/, const VertId& /*cOpposit*/ ) const
{
    Vector3d aP = Vector3d( points[a] );
    Vector3d bP = Vector3d( points[b] );
    Vector3d cP = Vector3d( points[c] );
    if ( dot( norm, cross( bP - aP, cP - aP ) ) < 0.0 )
        return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

    return circumcircleDiameter( aP, bP, cP );
}

double PlaneFillMetric::getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const
{
    return 0.0;
}

PlaneFillMetric::PlaneFillMetric( const Mesh& mesh, EdgeId e0 ) : 
    points{mesh.points}
{
    norm = Vector3d();
    for ( auto e : leftRing( mesh.topology, e0 ) )
    {
        norm += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    }
    norm = norm.normalized();
    hasEdgeMetric = false;
}

PlaneNormalizedFillMetric::PlaneNormalizedFillMetric( const Mesh& mesh, EdgeId e0 ) :
    points{ mesh.points }
{
    norm = Vector3d();
    for ( auto e : leftRing( mesh.topology, e0 ) )
    {
        norm += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    }
    norm = norm.normalized();
    hasEdgeMetric = false;
}

double PlaneNormalizedFillMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const
{
    Vector3d aP = Vector3d( points[a] );
    Vector3d bP = Vector3d( points[b] );
    Vector3d cP = Vector3d( points[c] );

    if ( dot( norm, cross( bP - aP, cP - aP ) ) < 0.0 )
        return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

    auto ar = triangleAspectRatio( aP, bP, cP );
    if ( ar > BadTriangulationMetric )
        return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

    return circumcircleDiameter( aP, bP, cP ) * ar;
}

double PlaneNormalizedFillMetric::getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const
{
    return 0.0;
}

ComplexStitchMetric::ComplexStitchMetric( const Mesh& mesh ) :
    points{ mesh.points }
{
    hasEdgeMetric = false;
}

double ComplexStitchMetric::getEdgeMetric( const VertId& a, const VertId& b, const VertId& l, const VertId& r ) const
{
    auto ab = ( points[b] - points[a] );
    auto normL = cross( points[l] - points[a], ab ).normalized();
    auto normR = cross( ab, points[r] - points[a] ).normalized();
    return ( 1.0 - dot( normL, normR ) ) * 1e4; // 1e4 because aspect ratio grows infinitely, and we need more affect from angles
}

double ComplexStitchMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const
{
    auto ratio = ( triangleAspectRatio( points[a], points[b], points[c] ) - 1.0f ) * 1e-2f; // 1e-2 because aspect ratio grows infinitely
    auto cc = cOpposit.valid() ? getEdgeMetric( a, b, c, cOpposit ) : 0.0;
    auto aa = aOpposit.valid() ? getEdgeMetric( b, c, a, aOpposit ) : 0.0;
    return ratio + cc + aa;
}

double EdgeLengthFillMetric::getEdgeMetric( const VertId& a, const VertId& b, const VertId&, const VertId& ) const
{
    // edge metric is called when merge 2 parts of triangulation is performed
    // this metric already counted ab edge weight in both parts, so we need to subtract it to be correct
    return -( points[b] - points[a] ).length();
}

double EdgeLengthFillMetric::getTriangleMetric( const VertId& a, const VertId&, const VertId& c, const VertId&, const VertId& ) const
{
    // ac is new edge
    return ( points[c] - points[a] ).length();
}

CircumscribedStitchMetric::CircumscribedStitchMetric( const Mesh& mesh ):
    points{ mesh.points }
{
    hasEdgeMetric = false;
}

double CircumscribedStitchMetric::getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const
{
    return 0.0;
}

double CircumscribedStitchMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const
{
    return circumcircleDiameter( points[a], points[b], points[c] );
}

EdgeLengthStitchMetric::EdgeLengthStitchMetric( const Mesh& mesh ) :
    points{ mesh.points }
{
    hasEdgeMetric = false;
}

double EdgeLengthStitchMetric::getTriangleMetric( const VertId& a, const VertId&, const VertId& c, const VertId&, const VertId& ) const
{
    // ac is new edge
    return ( points[c] - points[a] ).length();
}

double EdgeLengthStitchMetric::getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const
{
    return 0.0;
}

VerticalStitchMetric::VerticalStitchMetric( const Mesh& mesh, const Vector3f& upDir ):
    points{ mesh.points },
    upDirection{ upDir.normalized() }
{
    hasEdgeMetric = false;
}

double VerticalStitchMetric::getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const
{
    auto ab = points[b] - points[a];
    auto ac = points[c] - points[a];

    auto norm = cross( ab, ac ); // dbl area
    auto parallelPenalty = std::abs( dot( upDirection, norm ) );

    // sqr penalty and sides length to have valid m^4 power of each argument
    // norm.lengthSq - dbl area Sq - m^4
    // parallelPenaltySq ~ area cos(angle(updir,norm)) sq - m^4 
    // side length sq sq - m^4
    return 
        norm.lengthSq() + 
        sqr( parallelPenalty ) + 
        sqr( ab.lengthSq() + ac.lengthSq() + ( points[c] - points[b] ).lengthSq() ) * 0.5f;

}

double VerticalStitchMetric::getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const
{
    return 0.0;
}


}
