#include "MRMeshMeshDistance.h"
#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRTimer.h"
#include "MRMeshCollide.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriangleIntersection.h"

namespace MR
{

MeshMeshDistanceResult findDistance( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float upDistLimitSq )
{
    MR_TIMER;

    const AABBTree& aTree = a.mesh.getAABBTree();
    const AABBTree& bTree = b.mesh.getAABBTree();

    MeshMeshDistanceResult res;
    res.distSq = upDistLimitSq;
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
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
        NoInitNodeId a, b;
        float distSq;
    };
    // every split adds at most MaxChildren - 1 tasks, and two trees of 2^31 triangles have 32 levels together
    InplaceStack<SubTask, 128> subtasks;

    auto bBox = [&]( const Box3f & box ) { return rigidB2A ? transformed( box, *rigidB2A ) : box; };
    auto getSubTask = [&]( NodeId an, NodeId bn )
    {
        return SubTask { an, bn, aTree.nodes()[an].box.getDistanceSq( bBox( bTree.nodes()[bn].box ) ) };
    };

    if ( ( !a.region || aNodes.test( NodeId( 0 ) ) ) && ( !b.region || bNodes.test( NodeId( 0 ) ) ) )
        subtasks.push( getSubTask( NodeId( 0 ), NodeId( 0 ) ) );
    // the closest triangles found so far, their points are computed at the end
    TriTriDistanceResultf bestTd;
    Triangle3f bestAv, bestBv;

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( !( s.distSq < res.distSq ) )
            continue;

        const auto& aNode = aTree.nodes()[s.a];
        const auto& bNode = bTree.nodes()[s.b];
        if ( aNode.leaf() && bNode.leaf() )
        {
            Box3f bBoxes[AABBTree::MaxLeafTris];
            Triangle3f bTris[AABBTree::MaxLeafTris];
            for ( int j = 0; j < bNode.numLeafTris(); ++j )
            {
                const auto & bt = bTree.leafTris()[bNode.first + j];
                bBoxes[j] = bBox( bTree.leafBoxes()[bNode.first + j] );
                for ( int k = 0; k < 3; ++k )
                    bTris[j][k] = rigidB2A ? ( *rigidB2A )( b.mesh.points[bt.verts[k]] ) : b.mesh.points[bt.verts[k]];
            }
            for ( int i = aNode.first, aEnd = aNode.first + aNode.numLeafTris(); i < aEnd; ++i )
            {
                const auto & at = aTree.leafTris()[i];
                if ( a.region && !a.region->test( at.face ) )
                    continue;
                const auto & aBox = aTree.leafBoxes()[i];
                const Triangle3f av{ a.mesh.points[at.verts[0]], a.mesh.points[at.verts[1]], a.mesh.points[at.verts[2]] };
                for ( int j = 0; j < bNode.numLeafTris(); ++j )
                {
                    const auto bFace = bTree.leafTris()[bNode.first + j].face;
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
        SubTask ch[AABBTree::MaxChildren];
        int numCh = 0;
        auto add = [&]( const SubTask & t )
        {
            if ( !( t.distSq < res.distSq ) )
                return;
            int k = numCh++;
            for ( ; k > 0 && ch[k - 1].distSq < t.distSq; --k )
                ch[k] = ch[k - 1];
            ch[k] = t;
        };
        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            for ( int i = 0; i < aNode.count; ++i )
                if ( const NodeId c( aNode.first + i ); !a.region || aNodes.test( c ) )
                    add( getSubTask( c, s.b ) );
        }
        else
        {
            assert( !bNode.leaf() );
            for ( int i = 0; i < bNode.count; ++i )
                if ( const NodeId c( bNode.first + i ); !b.region || bNodes.test( c ) )
                    add( getSubTask( s.a, c ) );
        }
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
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
        res.a.mtp = MeshTriPoint( a.mesh.topology.edgeWithLeft( res.a.proj.face ), aBary );
        res.a.distSq = res.distSq;
        res.b.proj.point = rigidB2A ? rigidB2A->inverse()( bPt ) : bPt;
        res.b.mtp = MeshTriPoint( b.mesh.topology.edgeWithLeft( res.b.proj.face ), bBary );
        res.b.distSq = res.distSq;
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
            res.projectons[id] = std::make_pair( projectRes.mtp, distance );

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
    auto vertProj = [&]( const Mesh& mesh ) -> MeshProjectionResult
    {
        for ( auto e : orgRing( mesh.topology, bestV ) )
            if ( auto f = mesh.topology.left( e ) )
                return { .proj = { f, mesh.points[bestV] }, .mtp = { e, {} }, .distSq = sqr( bestDist ) };
        return {};
    };
    // the projection of the vertex on the other mesh
    auto otherProj = [&]( const Mesh& mesh, const MeshTriPoint& mtp ) -> MeshProjectionResult
    {
        return { .proj = { mesh.topology.left( mtp.e ), mesh.triPoint( mtp ) }, .mtp = mtp, .distSq = sqr( bestDist ) };
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
    bool isBOutside = a.mesh.isOutsideByProjNorm( rigidB2A ? ( *rigidB2A )( distRes.b.proj.point ) : distRes.b.proj.point, distRes.a, a.region );
    bool isAOutside = b.mesh.isOutsideByProjNorm( rigidB2A ? rigidB2A->inverse()( distRes.a.proj.point ) : distRes.a.proj.point, distRes.b, b.region );
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

float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    const auto& bMeshVerts = b.mesh.points;
    auto vertBitSet = getIncidentVerts( b.mesh.topology, b.mesh.topology.getFaceIds( b.region ) );
    if ( !vertBitSet.any() )
        return 0.0f;

    return tbb::parallel_reduce
    (
        tbb::blocked_range( vertBitSet.find_first(), vertBitSet.find_last() + 1 ),
        0.0f,
        [&] ( const tbb::blocked_range<VertId>& range, float init )
        {
        for ( VertId i = range.begin(); i < range.end(); ++i )
        {
            if ( !vertBitSet.test( i ) )
                continue;

            auto distSq = findProjection( rigidB2A ? (*rigidB2A)( bMeshVerts[i] ) : bMeshVerts[i], a, maxDistanceSq ).distSq;
            if ( distSq > init )
                init = distSq;
        }

        return  init;
        },
        [] ( float a, float b ) -> float
        {
            return a > b ? a : b;
        }
    );
}

float findMaxDistanceSq( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::unique_ptr<AffineXf3f> rigidA2B = rigidB2A ? std::make_unique<AffineXf3f>( rigidB2A->inverse() ) : nullptr;
    return std::max( findMaxDistanceSqOneWay( a, b, rigidB2A, maxDistanceSq ), findMaxDistanceSqOneWay( b, a, rigidA2B.get(), maxDistanceSq ) );
}

} //namespace MR
