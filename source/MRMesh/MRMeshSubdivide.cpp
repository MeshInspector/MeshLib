#include "MRMeshSubdivide.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRMeshDelone.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include <queue>
#include "MRGTest.h"

namespace MR
{

struct EdgeLength
{
    UndirectedEdgeId edge;
    float lenSq = 0; // at the moment the edge was put in the queue

    EdgeLength() = default;
    EdgeLength( UndirectedEdgeId edge, float lenSq ) : edge( edge ), lenSq( lenSq ) {}

    auto asPair() const { return std::make_pair( lenSq, edge ); }
};

inline bool operator < ( const EdgeLength & a, const EdgeLength & b )
{
    return a.asPair() < b.asPair();
}

int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings )
{
    MR_TIMER;
    MR_MESH_WRITER( mesh );

    const float maxEdgeLenSq = sqr( settings.maxEdgeLen );
    std::priority_queue<EdgeLength> queue;

    // region is changed during subdivision,
    // so if it has invalid faces (they can become valid later) some collisions can occur
    // better to filter valid faces in first step
    if ( settings.region )
        *settings.region &= mesh.topology.getValidFaces();

    auto addInQueue = [&]( UndirectedEdgeId e )
    {
        bool canSubdivide = settings.subdivideBorder ? mesh.topology.isInnerOrBdEdge( e, settings.region ) : mesh.topology.isInnerEdge( e, settings.region );
        if ( !settings.subdivideBorder && canSubdivide )
        {
            EdgeId eDir = EdgeId( e << 1 );
            auto f = mesh.topology.left( eDir );
            Vector3f vp[3];
            if ( f.valid() )
            {
                mesh.getTriPoints( f, vp[0], vp[1], vp[2] );
                canSubdivide = triangleAspectRatio( vp[0], vp[1], vp[2] ) < settings.critAspectRatio;
            }
            if ( canSubdivide )
            {
                f = mesh.topology.right( eDir );
                if ( f.valid() )
                {
                    mesh.getTriPoints( f, vp[0], vp[1], vp[2] );
                    canSubdivide = triangleAspectRatio( vp[0], vp[1], vp[2] ) < settings.critAspectRatio;
                }
            }
        }
        if ( !canSubdivide )
            return;
        float lenSq = mesh.edgeLengthSq( e );
        if ( lenSq < maxEdgeLenSq )
            return;
        queue.emplace( e, lenSq );
    };

    for ( UndirectedEdgeId e : undirectedEdges( mesh.topology ) )
    {
        addInQueue( e );
    }

    int splitsDone = 0;
    while ( splitsDone < settings.maxEdgeSplits && !queue.empty() )
    {
        auto el = queue.top();
        queue.pop();
        if ( el.lenSq != mesh.edgeLengthSq( el.edge ) )
            continue; // outdated record in the queue
        auto newVertId = mesh.splitEdge( el.edge, settings.region );
        if ( settings.newVerts )
            settings.newVerts->autoResizeSet( newVertId );
        if ( settings.onVertCreated )
            settings.onVertCreated( newVertId );
        ++splitsDone;
        makeDeloneOriginRing( mesh, el.edge, settings.maxDeviationAfterFlip, settings.region );
        for ( auto e : orgRing( mesh.topology, EdgeId{ el.edge } ) )
            addInQueue( e.undirected() );
    }

    return splitsDone;
}

TEST(MRMesh, SplitEdge) 
{
    std::vector<VertId> v{ 
        VertId{0}, VertId{1}, VertId{2}, 
        VertId{0}, VertId{2}, VertId{3}
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromVertexTriples( v );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    EXPECT_EQ( mesh.topology.numValidVerts(), 4 );
    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(9) ); // 5*2 = 10 half-edges in total

    FaceBitSet region( 2 );
    region.set( 0_f );

    auto e02 = mesh.topology.findEdge( VertId{0}, VertId{2} );
    EXPECT_TRUE( e02.valid() );
    VertId v02 = mesh.splitEdge( e02, &region );
    EXPECT_EQ( mesh.topology.numValidVerts(), 5 );
    EXPECT_EQ( mesh.points.size(), 5 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(15) ); // 8*2 = 16 half-edges in total
    EXPECT_EQ( mesh.points[v02], ( Vector3f(.5f, .5f, 0.f) ) );
    EXPECT_EQ( region.count(), 2 );

    auto e01 = mesh.topology.findEdge( VertId{0}, VertId{1} );
    EXPECT_TRUE( e01.valid() );
    VertId v01 = mesh.splitEdge( e01, &region );
    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.points.size(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 5 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(19) ); // 10*2 = 20 half-edges in total
    EXPECT_EQ( mesh.points[v01], ( Vector3f(.5f, 0.f, 0.f) ) );
    EXPECT_EQ( region.count(), 3 );
}

TEST(MRMesh, SubdivideMesh) 
{
    std::vector<VertId> v{ 
        VertId{0}, VertId{1}, VertId{2}, 
        VertId{0}, VertId{2}, VertId{3}
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromVertexTriples( v );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    FaceBitSet region( 2 );
    region.set( 0_f );

    SubdivideSettings settings;
    settings.maxEdgeLen = 0.3f;
    settings.maxEdgeSplits = 1000;
    settings.maxDeviationAfterFlip = FLT_MAX;
    settings.region = &region;
    int splitsDone = subdivideMesh( mesh, settings );
    EXPECT_TRUE( splitsDone > 19 && splitsDone < 25 );
    EXPECT_TRUE( region.count() * 2 + 3 > mesh.topology.numValidFaces() );
    EXPECT_TRUE( region.count() * 2 - 3 > mesh.topology.numValidFaces() );

    settings.maxEdgeLen = 0.1f;
    settings.maxEdgeSplits = 10;
    splitsDone = subdivideMesh( mesh, settings );
    EXPECT_TRUE( splitsDone == 10 );
    EXPECT_TRUE( region.count() * 2 + 3 > mesh.topology.numValidFaces() );
    EXPECT_TRUE( region.count() * 2 - 3 > mesh.topology.numValidFaces() );
}

} // namespace MR
