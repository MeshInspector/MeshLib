#include "MRMeshMeshDistance.h"
#include "MRAABBTreeTraversal.hpp"
#include "MRInplaceStack.h"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRTimer.h"
#include "MRMeshCollide.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriangleIntersection.h"
#include <optional>

namespace MR
{

/// the point as the projection of the other point located at given squared distance
static MeshProjectionResult toProjection( const PointOnFaceBary& p, float distSq )
{
    return { p, p ? distSq : 0 };
}

MeshMeshDistanceResult findDistance( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float upDistLimitSq )
{
    MR_TIMER;

    const AABBTree& aTree = a.mesh.getAABBTree();
    const AABBTree& bTree = b.mesh.getAABBTree();

    MeshMeshDistanceResult res;
    res.distSq = upDistLimitSq;
    if ( aTree.empty() || bTree.empty() )
    {
        assert( false );
        return res;
    }

    NodeBitSet aNodes, bNodes;
    if ( a.region )
        aNodes = aTree.getNodesFromFaces( *a.region );
    if ( b.region )
        bNodes = bTree.getNodesFromFaces( *b.region );

    struct SubTask
    {
        WideNode a, b;
        float distSq;
    };
    // every split adds at most MaxChildren - 1 tasks, and two trees of 2^31 triangles have 32 levels together
    InplaceStack<SubTask, 128> subtasks;

    auto bBox = [&]( const Box3f & box ) { return rigidB2A ? transformed( box, *rigidB2A ) : box; };
    auto getSubTask = [&]( const WideNode & an, const WideNode & bn )
    {
        return SubTask { an, bn, an.box.getDistanceSq( bBox( bn.box ) ) };
    };

    if ( ( !a.region || aNodes.test( NodeId( 0 ) ) ) && ( !b.region || bNodes.test( NodeId( 0 ) ) ) )
        subtasks.push( getSubTask( rootWideNode( aTree ), rootWideNode( bTree ) ) );
    // the closest triangles found so far, their points are computed at the end
    TriTriDistanceResultf bestTd;
    Triangle3f bestAv, bestBv;

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( !( s.distSq < res.distSq ) )
            continue;

        if ( s.a.leaf() && s.b.leaf() )
        {
            const int aFirst = ~s.a.ref * AABBTree::MaxLeafTris;
            const int bFirst = ~s.b.ref * AABBTree::MaxLeafTris;
            const int bNum = bTree.numLeafTris( ~s.b.ref );
            const auto & bLeafBoxes = bTree.leafBoxes( ~s.b.ref );
            const auto & aLeafBoxes = aTree.leafBoxes( ~s.a.ref );
            Box3f bBoxes[AABBTree::MaxLeafTris];
            Triangle3f bTris[AABBTree::MaxLeafTris];
            for ( int j = 0; j < bNum; ++j )
            {
                const auto & bt = bTree.leafTris()[bFirst + j];
                bBoxes[j] = bBox( bLeafBoxes.get( j ) );
                // not a single line with rigidB2A ? xf( p ) : p: MSVC writes the transformed point to a temporary as three floats
                // and copies it out with an 8-byte load, a store-forwarding stall
                for ( int k = 0; k < 3; ++k )
                {
                    bTris[j][k] = b.mesh.points[bt.verts[k]];
                    if ( rigidB2A )
                        bTris[j][k] = ( *rigidB2A )( bTris[j][k] );
                }
            }
            for ( int i = aFirst, aEnd = aFirst + aTree.numLeafTris( ~s.a.ref ); i < aEnd; ++i )
            {
                const auto & at = aTree.leafTris()[i];
                if ( a.region && !a.region->test( at.face ) )
                    continue;
                const auto aBox = aLeafBoxes.get( i - aFirst );
                const Triangle3f av{ a.mesh.points[at.verts[0]], a.mesh.points[at.verts[1]], a.mesh.points[at.verts[2]] };
                for ( int j = 0; j < bNum; ++j )
                {
                    const auto bFace = bTree.leafTris()[bFirst + j].face;
                    if ( b.region && !b.region->test( bFace ) )
                        continue;
                    if ( !( aBox.getDistanceSq( bBoxes[j] ) < res.distSq ) )
                        continue;
                    const auto td = findTriTriDistance( av, bTris[j], { .upDistLimitSq = res.distSq, .upLimitCheck = UpLimitCheck::GreaterOrEqual } );
                    if ( td.distSq < res.distSq )
                    {
                        res.distSq = td.distSq;
                        res.a.proj.face = at.face;
                        res.b.proj.face = bFace;
                        bestTd = td;
                        bestAv = av;
                        bestBv = bTris[j];
                    }
                }
            }
            continue;
        }

        // split the bigger non-leaf node, and add the children with smaller distance last to descend there first
        const bool splitA = !s.a.leaf() && ( s.b.leaf() || s.a.box.volume() >= s.b.box.volume() );
        assert( splitA || !s.b.leaf() );
        const auto & tree = splitA ? aTree : bTree;
        const auto & sn = tree.simdNodes()[splitA ? s.a.ref : s.b.ref];
        // the distances to the children, computed for all of them at once as getSubTask does
        alignas( 16 ) float d[AABBTree::MaxChildren];
        if ( splitA )
            Simd4::store( d, boxBoxDistSq( sn, bBox( s.b.box ) ) );
        else if ( !rigidB2A )
            Simd4::store( d, boxBoxDistSq( sn, s.a.box ) );
        else
        {
            for ( int i = 0; i < AABBTree::MaxChildren; ++i )
                d[i] = sn.child[i] ? s.a.box.getDistanceSq( bBox( sn.get( i ) ) ) : FLT_MAX;
        }
        const auto * regionNodes = splitA ? ( a.region ? &aNodes : nullptr ) : ( b.region ? &bNodes : nullptr );
        int ch[AABBTree::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < AABBTree::MaxChildren && sn.child[i]; ++i )
        {
            if ( !( d[i] < res.distSq ) )
                continue;
            if ( regionNodes && !regionNodes->test( tree.nodeId( sn.child[i] ) ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && d[ch[k - 1]] < d[i]; --k )
                ch[k] = ch[k - 1];
            ch[k] = i;
        }
        for ( int k = 0; k < numCh; ++k )
        {
            const int i = ch[k];
            const auto c = childWideNode( sn, i );
            subtasks.push( splitA ? SubTask{ c, s.b, d[i] } : SubTask{ s.a, c, d[i] } );
        }
    }

    if ( res.a ) // found closer than upDistLimitSq
    {
        Vector3f aPt = bestTd.a;
        Vector3f bPt = bestTd.b;
        TriPointf aBary, bBary;
        if ( res.distSq == 0 )
        {
            auto pt = findTriangleTriangleIntersection( bestAv[0], bestAv[1], bestAv[2], bestBv[0], bestBv[1], bestBv[2] );
            if ( pt )
                aPt = bPt = *pt;
            else
            {
                // might fall back here in degenerated scenarios
                aPt = bPt = 0.5f * ( aPt + bPt );
            }
            aBary = TriPointf( aPt, bestAv[0], bestAv[1], bestAv[2] );
            bBary = TriPointf( bPt, bestBv[0], bestBv[1], bestBv[2] );
        }
        else
        {
            aBary = bestTd.aBary( bestAv );
            bBary = bestTd.bBary( bestBv );
        }
        res.a.proj.point = aPt;
        res.a.tp = aBary;
        res.b.proj.point = rigidB2A ? rigidB2A->inverse()( bPt ) : bPt;
        res.b.tp = bBary;
    }

    return res;
}

struct InternalZoneWithProjections
{
    VertBitSet vertBS;
    std::vector<std::pair<MeshTriPoint, float>> projectons;
};

InternalZoneWithProjections findSignedDistanceOneWay( const MeshPart & a, const MeshPart & b,
                                                      const std::vector<FaceFace>& collisions,
                                                      bool bIsRef = false,
                                                      const AffineXf3f* rigidB2A = nullptr )
{
    const auto& ref  = bIsRef ? b : a;
    const auto& test = bIsRef ? a : b;

    FaceBitSet refFaces;
    refFaces.resize( ref.mesh.topology.edgePerFace().size() );

    for ( const auto& ff : collisions )
        refFaces.set( bIsRef ? ff.bFace : ff.aFace );

    auto ref2Test = rigidB2A ? ( bIsRef ? *rigidB2A : rigidB2A->inverse() ) : AffineXf3f();
    VertBitSet refRegionVerts;
    if ( ref.region )
        refRegionVerts = getIncidentVerts( ref.mesh.topology, *ref.region );
    VertBitSet queue = getIncidentVerts( ref.mesh.topology, refFaces );

    InternalZoneWithProjections res;
    res.projectons.resize( ref.mesh.points.size(), { {}, 0.0f } );

    while ( queue.any() )
    {
        tbb::enumerable_thread_specific<std::vector<VertId>> threadData;
        BitSetParallelFor( queue, threadData, [&]( VertId id, auto& localData )
        {
            const auto point = rigidB2A ? ref2Test( ref.mesh.points[id] ) : ref.mesh.points[id];
            auto projectRes = findProjection( point, test.mesh );
            if ( !contains( test.region, projectRes.proj.face ) )
                return;
            const auto distance = test.mesh.signedDistance( point, projectRes );
            if ( distance > 0.0f )
                return;
            res.projectons[id] = std::make_pair( projectRes.mtp( test.mesh.topology ), distance );

            for ( EdgeId e : orgRing( ref.mesh.topology, id ) )
            {
                const auto v = ref.mesh.topology.dest( e );
                if ( !ref.region || refRegionVerts.test( v ) )
                    localData.push_back( v );
            }
        } );
        res.vertBS |= queue;
        queue.reset();
        for ( const auto& verts : threadData )
            for ( auto v : verts )
                queue.set( v );
        queue -= res.vertBS;
    }

    return res;
}

MeshMeshSignedDistanceResult findSignedDistance( const MeshPart & a, const MeshPart & b, const AffineXf3f* rigidB2A, float upDistLimitSq )
{
    MR_TIMER;
    auto res = findDistance( a, b, rigidB2A, upDistLimitSq > 0.0f ? upDistLimitSq : FLT_TRUE_MIN );
    if ( res.distSq == FLT_TRUE_MIN )
        res.distSq = 0.0f;
    if ( !res.a || !res.b )
    {   // findDistance returns the limit sentinel when nothing closer was found
        assert( res.distSq == upDistLimitSq );
        return { res.a, res.b, MeshMeshCollisionStatus::NotColliding, std::sqrt( res.distSq ) };
    }
    assert( res.distSq == 0.0f || res.distSq < upDistLimitSq );
    std::vector<FaceFace> collisions;
    auto status = findCollisionStatus( a, b, res, rigidB2A, &collisions );
    if ( status == MeshMeshCollisionStatus::Touching )
        return { res.a, res.b, status, 0 }; // two meshes touch one another but do not intersect
    if ( status != MeshMeshCollisionStatus::Colliding )
    {
        auto dist = status == MeshMeshCollisionStatus::BothOutside ? std::sqrt( res.distSq ) : -std::sqrt( res.distSq );
        return { res.a, res.b, status, dist };
    }

    // colliding

    auto zoneAndDistancesAB = findSignedDistanceOneWay( a, b, collisions, false, rigidB2A );
    auto zoneAndDistancesBA = findSignedDistanceOneWay( a, b, collisions, true, rigidB2A );

    // the vertex of one mesh deepest inside the other one
    VertId bestV;
    bool bestInB = false;
    float bestDist = FLT_MAX;
    for ( VertId id : zoneAndDistancesAB.vertBS )
    {
        const auto& [mtp, dist] = zoneAndDistancesAB.projectons[id];
        if ( mtp.valid() && dist < bestDist )
        {
            bestV = id;
            bestDist = dist;
        }
    }
    for ( VertId id : zoneAndDistancesBA.vertBS )
    {
        const auto& [mtp, dist] = zoneAndDistancesBA.projectons[id];
        if ( mtp.valid() && dist < bestDist )
        {
            bestV = id;
            bestInB = true;
            bestDist = dist;
        }
    }
    if ( bestDist > 0.0f )
        return { res.a, res.b, status, 0.0f };

    // the vertex as a point of one of its faces
    auto vertProj = [&]( const Mesh& mesh ) -> PointOnFaceBary
    {
        for ( auto e : orgRing( mesh.topology, bestV ) )
            if ( auto f = mesh.topology.left( e ) )
                return { .proj = { f, mesh.points[bestV] }, .tp = MeshTriPoint( e, {} ).canonical( mesh.topology ).bary };
        return {};
    };
    // the projection of the vertex on the other mesh
    auto otherProj = [&]( const Mesh& mesh, const MeshTriPoint& mtp ) -> PointOnFaceBary
    {
        return { .proj = { mesh.topology.left( mtp.e ), mesh.triPoint( mtp ) }, .tp = mtp.canonical( mesh.topology ).bary };
    };
    if ( bestInB )
        return { otherProj( a.mesh, zoneAndDistancesBA.projectons[bestV].first ), vertProj( b.mesh ), status, bestDist };
    return { vertProj( a.mesh ), otherProj( b.mesh, zoneAndDistancesAB.projectons[bestV].first ), status, bestDist };
}

MeshMeshCollisionStatus findCollisionStatus( const MeshPart& a, const MeshPart& b, const MeshMeshDistanceResult& distRes, 
    const AffineXf3f* rigidB2A /*= nullptr*/, std::vector<FaceFace>* collisions /*= nullptr */ )
{
    MR_TIMER;
    MeshMeshCollisionStatus status;
    if ( distRes.distSq == 0.0f )
    {
        auto thisCollisions = findCollidingTriangles( a, b, rigidB2A );
        status = thisCollisions.empty() ? MeshMeshCollisionStatus::Touching : MeshMeshCollisionStatus::Colliding;
        if ( collisions )
            *collisions = std::move( thisCollisions );
        return status;
    }
    bool isBOutside = a.mesh.isOutsideByProjNorm( rigidB2A ? ( *rigidB2A )( distRes.b.proj.point ) : distRes.b.proj.point, toProjection( distRes.a, distRes.distSq ), a.region );
    bool isAOutside = b.mesh.isOutsideByProjNorm( rigidB2A ? rigidB2A->inverse()( distRes.a.proj.point ) : distRes.a.proj.point, toProjection( distRes.b, distRes.distSq ), b.region );
    if ( isBOutside && isAOutside )
        status = MeshMeshCollisionStatus::BothOutside;
    else if ( !isBOutside && !isAOutside )
        status = MeshMeshCollisionStatus::BothInside;
    else if ( isBOutside )
        status = MeshMeshCollisionStatus::AInside;
    else
        status = MeshMeshCollisionStatus::BInside;
    return status;
}

MeshMeshCollisionStatus findCollisionStatus( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    MR_TIMER;
    auto res = findDistance( a, b, rigidB2A );
    return findCollisionStatus( a, b, res, rigidB2A );
}

MeshMeshDistanceResult findMaxDistanceOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    MeshMeshDistanceResult res;
    const auto& bMeshVerts = b.mesh.points;
    const auto& bFaces = b.mesh.topology.getFaceIds( b.region );
    auto vertBitSet = getIncidentVerts( b.mesh.topology, bFaces );
    if ( !vertBitSet.any() )
        return res;

    auto inA = [&]( VertId v ) { return rigidB2A ? ( *rigidB2A )( bMeshVerts[v] ) : bMeshVerts[v]; };

    struct VertDist
    {
        VertId v;
        float distSq = -1;
    };
    // of several equally distant vertices, the one with the smallest id is taken
    const auto best = tbb::parallel_reduce
    (
        tbb::blocked_range( vertBitSet.find_first(), vertBitSet.find_last() + 1 ),
        VertDist{},
        [&] ( const tbb::blocked_range<VertId>& range, VertDist init )
        {
            for ( VertId i = range.begin(); i < range.end(); ++i )
            {
                if ( !vertBitSet.test( i ) )
                    continue;
                const auto distSq = findProjection( inA( i ), a, maxDistanceSq ).distSq;
                if ( distSq > init.distSq )
                    init = { i, distSq };
            }
            return init;
        },
        [] ( const VertDist& x, const VertDist& y ) { return y.distSq > x.distSq ? y : x; }
    );

    const auto proj = findProjection( inA( best.v ), a, maxDistanceSq );
    res.a = proj;
    res.distSq = proj.distSq;
    // the vertex as a point of one of its triangles
    for ( auto e : orgRing( b.mesh.topology, best.v ) )
    {
        if ( auto f = b.mesh.topology.left( e ); f && bFaces.test( f ) )
        {
            res.b = { .proj = { f, bMeshVerts[best.v] }, .tp = MeshTriPoint( e, {} ).canonical( b.mesh.topology ).bary };
            break;
        }
    }
    return res;
}

MeshMeshDistanceResult findMaxDistance( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::optional<AffineXf3f> rigidA2B;
    if ( rigidB2A )
        rigidA2B = rigidB2A->inverse();
    auto res = findMaxDistanceOneWay( a, b, rigidB2A, maxDistanceSq );
    const auto ba = findMaxDistanceOneWay( b, a, rigidA2B ? &*rigidA2B : nullptr, maxDistanceSq );
    if ( ba.distSq > res.distSq )
        res = { ba.b, ba.a, ba.distSq };
    return res;
}

float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    return findMaxDistanceOneWay( a, b, rigidB2A, maxDistanceSq ).distSq;
}

float findMaxDistanceSq( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    return findMaxDistance( a, b, rigidB2A, maxDistanceSq ).distSq;
}

} //namespace MR
