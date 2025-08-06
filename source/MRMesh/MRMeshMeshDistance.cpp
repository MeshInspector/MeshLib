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
#include "MRPch/MRTBB.h"

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
    NodeBitSet* aNodesPtr{nullptr}, *bNodesPtr{nullptr};
    if ( a.region )
    {
        aNodes = aTree.getNodesFromLeaves( *a.region );
        aNodesPtr = &aNodes;
    }
    if ( b.region )
    {
        bNodes = bTree.getNodesFromLeaves( *b.region );
        bNodesPtr = &bNodes;
    }

    struct SubTask
    {
        NoInitNodeId a, b;
        float distSq;
    };
    InplaceStack<SubTask, 128> subtasks;

    auto addSubTask = [&]( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&]( NodeId a, NodeId b )
    {
        float distSq = aTree.nodes()[a].box.getDistanceSq( transformed( bTree.nodes()[b].box, rigidB2A ) );
        return SubTask { a, b, distSq };
    };

    addSubTask( getSubTask( aTree.rootNodeId(), bTree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( aNodesPtr && !aNodes.test( s.a ) )
            continue;
        if ( bNodesPtr && !bNodes.test( s.b ) )
            continue;

        const auto& aNode = aTree[s.a];
        const auto& bNode = bTree[s.b];
        if ( s.distSq >= res.distSq )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            const auto aFace = aNode.leafId();
            const auto bFace = bNode.leafId();

            Vector3f aPt, bPt;
            Vector3f av[3], bv[3];
            a.mesh.getTriPoints( aFace, av[0], av[1], av[2] );
            b.mesh.getTriPoints( bFace, bv[0], bv[1], bv[2] );
            if ( rigidB2A )
            {
                bv[0] = ( *rigidB2A )( bv[0] );
                bv[1] = ( *rigidB2A )( bv[1] );
                bv[2] = ( *rigidB2A )( bv[2] );
            }

            float distSq = TriDist( aPt, bPt, av, bv );
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.a.point = aPt;
                res.a.face = aNode.leafId();
                res.b.point = bPt;
                res.b.face = bNode.leafId();
            }
            continue;
        }

        SubTask s1, s2;
        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            s1 = getSubTask( aNode.l, s.b );
            s2 = getSubTask( aNode.r, s.b );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            s1 = getSubTask( s.a, bNode.l );
            s2 = getSubTask( s.a, bNode.r );
        }
        // add task with smaller distance last to descend there first
        if ( s1.distSq < s2.distSq )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
    }

    if ( rigidB2A && res.distSq < upDistLimitSq )
        res.b.point = rigidB2A->inverse()( res.b.point );

    return res;
}

struct InternalZoneWithProjections
{
    VertBitSet vertBS;
    std::vector<std::pair<PointOnFace, float>> projectons;
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
            res.projectons[id] = std::make_pair( projectRes.proj, distance );

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

    // If meshes has no collision no need to find signed distance
    auto res = findDistance( a, b, rigidB2A, upDistLimitSq );
    if ( res.distSq > 0.0f )
        return { res.a, res.b, std::sqrt( res.distSq ) };

    auto collisions = findCollidingTriangles( a, b, rigidB2A );
    if ( collisions.empty() )
        return { res.a, res.b, 0 }; // two meshes touch one another but do not intersect

    auto zoneAndDistancesAB = findSignedDistanceOneWay( a, b, collisions, false, rigidB2A );
    auto zoneAndDistancesBA = findSignedDistanceOneWay( a, b, collisions, true, rigidB2A );

    MeshMeshSignedDistanceResult signedRes;
    signedRes.signedDist = FLT_MAX;
    auto triZoneA = getInnerFaces( a.mesh.topology, zoneAndDistancesAB.vertBS );
    auto triZoneB = getInnerFaces( b.mesh.topology, zoneAndDistancesBA.vertBS );

    auto getTriByVert = [&]( const MeshTopology& topology, VertId id )->FaceId
    {
        for ( auto e : orgRing( topology, id ) )
            if ( auto f = topology.left( e ) )
                return f;
        return FaceId{};
    };

    for ( VertId id : zoneAndDistancesAB.vertBS )
    {
        const auto& [proj, dist] = zoneAndDistancesAB.projectons[id];
        if ( !triZoneB.test( proj.face ) )
            continue;
        if ( dist < signedRes.signedDist )
        {
            signedRes.a = { getTriByVert( a.mesh.topology,id ), a.mesh.points[id] };
            signedRes.b = proj;
            signedRes.signedDist = dist;
        }
    }

    for ( VertId id : zoneAndDistancesBA.vertBS )
    {
        const auto& [proj, dist] = zoneAndDistancesBA.projectons[id];
        if ( !triZoneA.test( proj.face ) )
            continue;
        if ( dist < signedRes.signedDist )
        {
            signedRes.a = proj;
            signedRes.b = { getTriByVert( b.mesh.topology,id ), b.mesh.points[id] };
            signedRes.signedDist = dist;
        }
    }
    return (signedRes.signedDist > 0.0f) ? MeshMeshSignedDistanceResult{res.a, res.b, 0.0f} : signedRes;
}

MRMESH_API float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
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
