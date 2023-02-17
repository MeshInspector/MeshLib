#include "MRMeshSubdivide.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRMeshDelone.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include "MRGTest.h"
#include "MRPositionVertsSmoothly.h"
#include "MRRegionBoundary.h"
#include <queue>

namespace MR
{

struct EdgeLength
{
    UndirectedEdgeId edge;
    float lenSq = 0; // at the moment the edge was put in the queue

    EdgeLength() = default;
    EdgeLength( UndirectedEdgeId edge, float lenSq ) : edge( edge ), lenSq( lenSq ) {}
};

inline bool operator < ( const EdgeLength & a, const EdgeLength & b )
{
    return std::tie( a.lenSq, a.edge ) < std::tie( b.lenSq, b.edge );
}

int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings )
{
    MR_TIMER;

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

    if ( settings.progressCallback && !settings.progressCallback( 0.25f ) )
        return 0;

    MR_WRITER( mesh );

    int splitsDone = 0;
    int lastProgressSplitsDone = 0;
    VertBitSet newVerts;
    const float whileProgress = settings.smoothMode ? 0.5f : 0.75f;
    while ( splitsDone < settings.maxEdgeSplits && !queue.empty() )
    {
        if ( settings.progressCallback && splitsDone >= 1000 + lastProgressSplitsDone ) 
        {
            if ( !settings.progressCallback( 0.25f + whileProgress * splitsDone / settings.maxEdgeSplits ) )
                return splitsDone;
            lastProgressSplitsDone = splitsDone;
        }

        const auto el = queue.top();
        const EdgeId e = el.edge;
        queue.pop();

        if ( el.lenSq != mesh.edgeLengthSq( e ) )
            continue; // outdated record in the queue

        const auto e1 = mesh.splitEdge( e, mesh.edgeCenter( e ), settings.region );
        const auto newVertId = mesh.topology.org( e );

        if ( settings.smoothMode )
            newVerts.autoResizeSet( newVertId );
        if ( settings.newVerts )
            settings.newVerts->autoResizeSet( newVertId );
        if ( settings.onVertCreated )
            settings.onVertCreated( newVertId );
        if ( settings.onEdgeSplit )
            settings.onEdgeSplit( e1, e );
        ++splitsDone;
        makeDeloneOriginRing( mesh, e, {
            .maxDeviationAfterFlip = settings.maxDeviationAfterFlip,
            .maxAngleChange = settings.maxAngleChangeAfterFlip,
            .criticalTriAspectRatio = 1000,
            .region = settings.region,
            .notFlippable = settings.notFlippable } );
        for ( auto ei : orgRing( mesh.topology, e ) )
            addInQueue( ei.undirected() );
    }

    if ( settings.smoothMode )
    {
        if ( settings.progressCallback && !settings.progressCallback( 0.75f ) )
            return 0;
        if ( settings.minSharpDihedralAngle < PI )
        {
            const UndirectedEdgeBitSet creaseUEdges = mesh.findCreaseEdges( settings.minSharpDihedralAngle );
            if ( settings.progressCallback && !settings.progressCallback( 0.76f ) )
                return 0;
            const auto sharpVerts = getIncidentVerts( mesh.topology, creaseUEdges );
            if ( settings.progressCallback && !settings.progressCallback( 0.77f ) )
                return 0;
            positionVertsSmoothly( mesh, newVerts, Laplacian::EdgeWeights::Cotan, &sharpVerts );
        }
        else
            positionVertsSmoothly( mesh, newVerts, Laplacian::EdgeWeights::Cotan );
    }

    return splitsDone;
}

TEST(MRMesh, SubdivideMesh) 
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
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
