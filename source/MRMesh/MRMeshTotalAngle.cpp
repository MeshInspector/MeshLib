#include "MRMeshTotalAngle.h"
#include "MRMeshDelone.h"
#include "MRMesh.h"
#include "MRTriMath.h"
#include <cfloat>

namespace MR
{

namespace
{

// given quadrangle ABCD, computes AC * |angle(AC)|
float calcAngleLength( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    const auto ln = dirDblArea( a, c, d );
    const auto rn = dirDblArea( a, b, c );
    const auto ac = c - a;
    return ac.length() * std::abs( MR::dihedralAngle( ln, rn, ac ) );
}

} //anonymous namespace

float totalAngleIncreaseOnFlip( const MeshTopology & topology, const VertCoords & points, EdgeId edge, const FlipRegion & r )
{
    const auto can = canFlipEdge( topology, edge, r.region, r.notFlippable, r.vertRegion );
    if ( can == FlipEdge::Cannot )
        return FLT_MAX;
    if ( can == FlipEdge::Must )
        return -FLT_MAX;

    VertId a, b, c, d;
    topology.getLeftTriVerts( edge, a, c, d );
    auto prevEdge = topology.prev( edge );
    b = topology.dest( prevEdge );

    auto ap = points[a];
    auto bp = points[b];
    auto cp = points[c];
    auto dp = points[d];
    
    float res = calcAngleLength( bp, cp, dp, ap ) - calcAngleLength( ap, bp, cp, dp );
    if ( topology.right( prevEdge ) )
    {
        auto ep = points[topology.dest( topology.prev( prevEdge ) )];
        res += calcAngleLength( ap, ep, bp, dp ) - calcAngleLength( ap, ep, bp, cp );
    }
    if ( auto e1 = topology.next( edge.sym() ); topology.left( e1 ) )
    {
        auto fp = points[topology.dest( topology.next( e1 ) )];
        res += calcAngleLength( bp, fp, cp, dp ) - calcAngleLength( bp, fp, cp, ap );
    }
    if ( auto e2 = topology.prev( edge.sym() ); topology.right( e2 ) )
    {
        auto gp = points[topology.dest( topology.prev( e2 ) )];
        res += calcAngleLength( cp, gp, dp, bp ) - calcAngleLength( cp, gp, dp, ap );
    }
    if ( auto e3 = topology.next( edge ); topology.left( e3 ) )
    {
        auto hp = points[topology.dest( topology.next( e3 ) )];
        res += calcAngleLength( dp, hp, ap, bp ) - calcAngleLength( dp, hp, ap, cp );
    }

    return res;
}

} // namespace MR
