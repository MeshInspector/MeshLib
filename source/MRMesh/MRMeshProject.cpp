#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRClosestPointInTriangle.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq, FaceId skipFace )
{
    const AABBTree & tree = mp.mesh.getAABBTree();

    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
    {
        assert( false );
        return res;
    }

    struct SubTask
    {
        AABBTree::NodeId n;
        float distSq = 0;
        SubTask() = default;
        SubTask( AABBTree::NodeId n, float dd ) : n( n ), distSq( dd ) { }
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( AABBTree::NodeId n )
    {
        float distSq = ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).lengthSq();
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto & node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( face == skipFace )
                continue;
            if ( mp.region && !mp.region->test( face ) )
                continue;
            Vector3f a, b, c;
            mp.mesh.getTriPoints( face, a, b, c );
            if ( xf )
            {
                a = (*xf)( a );
                b = (*xf)( b );
                c = (*xf)( c );
            }
            
            // compute the closest point in double-precision, because float might be not enough
            const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
            const Vector3f proj( projD );
            const TriPointf bary( baryD );

            float distSq = ( proj - pt ).lengthSq();
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.proj.point = proj;
                res.proj.face = face;
                res.mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( face ), bary };
                if ( distSq <= loDistLimitSq )
                    break;
            }
            continue;
        }
        
        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.distSq < s2.distSq )
            std::swap( s1, s2 );
        assert ( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }

    return res;
}

std::optional<SignedDistanceToMeshResult> findSignedDistance( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq, float loDistLimitSq )
{
    auto projRes = findProjection( pt, mp, upDistLimitSq, nullptr, loDistLimitSq );
    std::optional<SignedDistanceToMeshResult> res;
    if ( !( projRes.distSq < upDistLimitSq ) || projRes.distSq < loDistLimitSq )
    {
        return res;
    }
    res = SignedDistanceToMeshResult();
    res->proj = projRes.proj;
    res->mtp = projRes.mtp;
    res->dist = mp.mesh.signedDistance( pt, projRes.mtp, mp.region );
    return res;
}

bool isInnerShellVert( const Mesh & mesh, const Vector3f & shellPoint, Side side )
{
    auto sd = findSignedDistance( shellPoint, mesh );
    assert( sd );
    if ( !sd )
        return false;
    if ( sd->mtp.isBd( mesh.topology ) )
        return false;
    if ( side == Side::Positive && sd->dist <= 0 )
        return false;
    if ( side == Side::Negative && sd->dist >= 0 )
        return false;
    return true;
}

VertBitSet findInnerShellVerts( const Mesh & mesh, const Mesh & shell, Side side )
{
    MR_TIMER
    VertBitSet res( shell.topology.vertSize() );
    BitSetParallelFor( shell.topology.getValidVerts(), [&]( VertId v )
    {
        if ( isInnerShellVert( mesh, shell.points[v], side ) )
            res.set( v );
    } );
    return res;
}

} //namespace MR
