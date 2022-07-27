#include "MRMeshDistance.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRTimer.h"
#include "MRUVSphere.h"
#include "MRMeshCollide.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

MeshDistanceResult findDistance( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float upDistLimitSq )
{
    MR_TIMER;

    const AABBTree& aTree = a.mesh.getAABBTree();
    const AABBTree& bTree = b.mesh.getAABBTree();

    MeshDistanceResult res;
    res.distSq = upDistLimitSq;
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
    {
        assert( false );
        return res;
    }

    AABBTree::NodeBitSet aNodes, bNodes;
    AABBTree::NodeBitSet* aNodesPtr{nullptr}, *bNodesPtr{nullptr};
    if ( a.region )
    {
        aNodes = aTree.getNodesFromFaces( *a.region );
        aNodesPtr = &aNodes;
    }
    if ( b.region )
    {
        bNodes = bTree.getNodesFromFaces( *b.region );
        bNodesPtr = &bNodes;
    }


    struct SubTask
    {
        AABBTree::NodeId a, b;
        float distSq = 0;
        SubTask() = default;
        SubTask( AABBTree::NodeId a, AABBTree::NodeId b, float dd ) : a( a ), b( b ), distSq( dd )
        {
        }
    };

    constexpr int MaxStackSize = 128; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( AABBTree::NodeId a, AABBTree::NodeId b )
    {
        float distSq = aTree.nodes()[a].box.getDistanceSq( transformed( bTree.nodes()[b].box, rigidB2A ) );
        return SubTask( a, b, distSq );
    };

    addSubTask( getSubTask( aTree.rootNodeId(), bTree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
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
        if ( s1.distSq < s2.distSq )
            std::swap( s1, s2 );
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
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

    while ( queue.count() != 0 )
    {
        tbb::enumerable_thread_specific<std::vector<VertId>> threadData;
        BitSetParallelFor( queue, [&]( VertId id )
        {
            const auto point = rigidB2A ? ref2Test( ref.mesh.points[id] ) : ref.mesh.points[id];
            auto projectRes = test.mesh.projectPoint( point );
            if ( !projectRes )
                return;
            if ( test.region && !test.region->test( projectRes->proj.face ) )
                return;
            const auto distance = test.mesh.signedDistance( point, projectRes->mtp );
            if ( distance > 0.0f )
                return;
            res.projectons[id] = std::make_pair( projectRes->proj, distance );

            auto& localData = threadData.local();
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

MeshSignedDistanceResult findSignedDistance( const MeshPart & a, const MeshPart & b, const AffineXf3f* rigidB2A, float upDistLimitSq )
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

    MeshSignedDistanceResult signedRes;
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
    return (signedRes.signedDist > 0.0f) ? MeshSignedDistanceResult{res.a, res.b, 0.0f} : signedRes;
}

MRMESH_API float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    const auto& bMeshVerts = b.mesh.points;
        
    return tbb::parallel_reduce
    (
        tbb::blocked_range( MR::begin(bMeshVerts), MR::end( bMeshVerts ) ),
        0.0f, 
        [&] ( const auto& range, float init )
        {
        for ( auto& vert : range )
        {
            auto distSq = findProjection( vert, a, maxDistanceSq, rigidB2A ).distSq;
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

MRMESH_API float findMaxDistanceSq( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::unique_ptr<AffineXf3f> rigidA2B = rigidB2A ? std::make_unique<AffineXf3f>( rigidB2A->inverse() ) : nullptr;
    return std::max( findMaxDistanceSqOneWay( a, b, rigidB2A, maxDistanceSq ), findMaxDistanceSqOneWay( b, a, rigidA2B.get(), maxDistanceSq ) );
}

TEST(MRMesh, MeshDistance) 
{
    Mesh sphere1 = makeUVSphere( 1, 8, 8 );

    auto d11 = findDistance( sphere1, sphere1, nullptr, FLT_MAX );
    EXPECT_EQ( d11.distSq, 0 );

    auto zShift = AffineXf3f::translation( Vector3f( 0, 0, 3 ) );
    auto d1z = findDistance( sphere1, sphere1, &zShift, FLT_MAX );
    EXPECT_EQ( d1z.distSq, 1 );

    Mesh sphere2 = makeUVSphere( 2, 8, 8 );

    auto d12 = findDistance( sphere1, sphere2, nullptr, FLT_MAX );
    float dist12 = std::sqrt( d12.distSq );
    EXPECT_TRUE( dist12 > 0.9f && dist12 < 1.0f );
}

} //namespace MR
