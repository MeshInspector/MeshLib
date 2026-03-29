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
    b = topology.dest( topology.prev( edge ) );

    auto ap = points[a];
    auto bp = points[b];
    auto cp = points[c];
    auto dp = points[d];
    
    float res = calcAngleLength( bp, cp, dp, ap ) - calcAngleLength( ap, bp, cp, dp );
    return res;
}

} // namespace MR
