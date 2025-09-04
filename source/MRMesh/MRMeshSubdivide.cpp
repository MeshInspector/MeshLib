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
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRBuffer.h"
#include "MRObjectMesh.h"
#include "MRMeshSubdivideCallbacks.h"
#include <queue>

namespace MR
{

struct EdgeLength
{
    UndirectedEdgeId edge;
    float lenSq; // at the moment the edge was put in the queue
    EdgeLength( UndirectedEdgeId edge = {}, float lenSq = 0 ) : edge( edge ), lenSq( lenSq ) {}
    EdgeLength( NoInit ) : edge( noInit ) {}
    explicit operator bool() const { return edge.valid(); }
};

inline bool operator < ( const EdgeLength & a, const EdgeLength & b )
{
    return std::tie( a.lenSq, a.edge ) < std::tie( b.lenSq, b.edge );
}

int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings )
{
    MR_TIMER;
    const float maxEdgeLenSq = sqr( settings.maxEdgeLen );
    Mesh original;
    if ( settings.projectOnOriginalMesh )
        original = mesh;

    // region is changed during subdivision,
    // so if it has invalid faces (they can become valid later) some collisions can occur
    // better to filter valid faces in first step
    if ( settings.region )
        *settings.region &= mesh.topology.getValidFaces();

    FaceBitSet aboveMaxTriAspectRatio, aboveMaxSplittableTriAspectRatio;
    if ( settings.maxTriAspectRatio >= 1 )
        aboveMaxTriAspectRatio.resize( mesh.topology.faceSize(), false );
    if ( settings.maxSplittableTriAspectRatio < FLT_MAX )
        aboveMaxSplittableTriAspectRatio.resize( mesh.topology.faceSize(), false );

    if ( !aboveMaxTriAspectRatio.empty() || !aboveMaxSplittableTriAspectRatio.empty() )
    {
        BitSetParallelFor( mesh.topology.getFaceIds( settings.region ), [&]( FaceId f )
        {
            const auto a = mesh.triangleAspectRatio( f );
            if ( !aboveMaxTriAspectRatio.empty() && a > settings.maxTriAspectRatio )
                aboveMaxTriAspectRatio.set( f );
            if ( !aboveMaxSplittableTriAspectRatio.empty() && a > settings.maxSplittableTriAspectRatio )
                aboveMaxSplittableTriAspectRatio.set( f );
        } );
    }
    auto numAboveMax = aboveMaxTriAspectRatio.count();

    auto getQueueElem = [&]( UndirectedEdgeId ue )
    {
        EdgeLength x;
        EdgeId e( ue );
        if ( settings.subdivideBorder ? !mesh.topology.isInnerOrBdEdge( e, settings.region )
                                      : !mesh.topology.isInnerEdge( e, settings.region ) )
            return x;
        const float lenSq = mesh.edgeLengthSq( e );
        if ( lenSq < maxEdgeLenSq )
            return x;
        if ( !aboveMaxSplittableTriAspectRatio.empty() )
        {
            if ( auto f = mesh.topology.left( e ); f && aboveMaxSplittableTriAspectRatio.test( f ) )
                return x;
            if ( auto f = mesh.topology.right( e ); f && aboveMaxSplittableTriAspectRatio.test( f ) )
                return x;
        }
        x.edge = ue;
        x.lenSq = lenSq;
        return x;
    };

    Vector<EdgeLength, UndirectedEdgeId> evec;
    evec.resizeNoInit( mesh.topology.undirectedEdgeSize() );
    ParallelFor( evec, [&]( UndirectedEdgeId ue )
    {
        EdgeLength x;
        if ( !mesh.topology.isLoneEdge( ue ) )
            x = getQueueElem( ue );
        evec[ue] = x;
    } );
    std::erase_if( evec.vec_, []( const EdgeLength & x ) { return !x; } );
    std::priority_queue<EdgeLength> queue( std::less<EdgeLength>(), std::move( evec.vec_ ) );

    if ( settings.progressCallback && !settings.progressCallback( 0.25f ) )
        return 0;

    MR_WRITER( mesh );

    int splitsDone = 0;
    int lastProgressSplitsDone = 0;
    VertBitSet newVerts;
    ProgressCallback notSmoothProgress = subprogress( settings.progressCallback, 0.25f, settings.smoothMode ? 0.75f : 1.0f );
    ProgressCallback whileProgress  =  subprogress( notSmoothProgress, 0.0f, settings.projectOnOriginalMesh ? 0.75f : 1.0f );
    while ( splitsDone < settings.maxEdgeSplits && !queue.empty() )
    {
        if ( settings.maxTriAspectRatio >= 1 && numAboveMax <= 0 )
            break;
        if ( settings.progressCallback && splitsDone >= 1000 + lastProgressSplitsDone )
        {
            if ( !whileProgress( float( splitsDone ) / settings.maxEdgeSplits ) )
                return 0;
            lastProgressSplitsDone = splitsDone;
        }

        const auto el = queue.top();
        const EdgeId e = el.edge;
        queue.pop();

        if ( el.lenSq != mesh.edgeLengthSq( e ) )
            continue; // outdated record in the queue

        if ( settings.beforeEdgeSplit && !settings.beforeEdgeSplit( e ) )
            continue;

        const auto e1 = mesh.splitEdge( e, mesh.edgeCenter( e ), settings.region );
        const auto newVertId = mesh.topology.org( e );

        // in smooth mode remember all new inner vertices to reposition them at the end
        if ( ( settings.smoothMode || settings.projectOnOriginalMesh ) && mesh.topology.left( e ) && mesh.topology.right( e ) )
            newVerts.autoResizeSet( newVertId );

        if ( settings.newVerts )
            settings.newVerts->autoResizeSet( newVertId );
        if ( settings.onVertCreated )
            settings.onVertCreated( newVertId );
        if ( settings.onEdgeSplit )
            settings.onEdgeSplit( e1, e );
        if ( settings.maintainRegion )
        {
            if ( contains( *settings.maintainRegion, mesh.topology.left( e ) ) )
                settings.maintainRegion->autoResizeSet( mesh.topology.left( e1 ) );
            if ( contains( *settings.maintainRegion, mesh.topology.right( e ) ) )
                settings.maintainRegion->autoResizeSet( mesh.topology.right( e1 ) );
        }
        if ( settings.notFlippable && settings.notFlippable->test( e.undirected() ) )
            settings.notFlippable->autoResizeSet( e1.undirected() );
        ++splitsDone;
        makeDeloneOriginRing( mesh, e, {
            .maxDeviationAfterFlip = settings.maxDeviationAfterFlip,
            .maxAngleChange = settings.maxAngleChangeAfterFlip,
            .criticalTriAspectRatio = settings.criticalAspectRatioFlip,
            .region = settings.region,
            .notFlippable = settings.notFlippable } );

        if ( !aboveMaxTriAspectRatio.empty() || !aboveMaxSplittableTriAspectRatio.empty() )
        {
            for ( auto ei : orgRing( mesh.topology, e ) )
            {
                const auto f = mesh.topology.left( ei );
                if ( !MR::contains( settings.region, f ) )
                    continue;
                const auto a = mesh.triangleAspectRatio( f );
                if ( !aboveMaxTriAspectRatio.empty() )
                {
                    const bool v = a > settings.maxTriAspectRatio;
                    if ( v != aboveMaxTriAspectRatio.autoResizeTestSet( f, v ) )
                        v ? ++numAboveMax : --numAboveMax;
                }
                if ( !aboveMaxSplittableTriAspectRatio.empty() )
                {
                    const bool v = a > settings.maxSplittableTriAspectRatio;
                    if ( v != aboveMaxSplittableTriAspectRatio.autoResizeTestSet( f, v ) && !v )
                        if ( auto x = getQueueElem( mesh.topology.prev( ei.sym() ) ) )
                            queue.push( std::move( x ) );
                }
            }
        }

        for ( auto ei : orgRing( mesh.topology, e ) )
            if ( auto x = getQueueElem( ei ) )
                queue.push( std::move( x ) );
    }

    if ( settings.projectOnOriginalMesh )
    {
        if ( !BitSetParallelFor( newVerts, [&]( VertId v )
        {
            mesh.points[v] = findProjection( mesh.points[v], original ).proj.point;
        }, subprogress( notSmoothProgress, 0.75f, 1.0f ) ) )
            return 0;
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
            positionVertsSmoothly( mesh, newVerts, EdgeWeights::Cotan, VertexMass::Unit, &sharpVerts );
        }
        else
            positionVertsSmoothly( mesh, newVerts, EdgeWeights::Cotan );
    }

    return splitsDone;
}

Expected<Mesh> copySubdividePackMesh( const MeshPart & mp, float voxelSize, const ProgressCallback & cb )
{
    MR_TIMER;
    Mesh subMesh;

    // copy mesh
    if ( mp.region )
        subMesh.addMeshPart( mp );
    else
        subMesh = mp.mesh;
    if ( !reportProgress( cb, 0.25f ) )
        return unexpectedOperationCanceled();

    // smaller edge lengths require too much space and time for subdivision,
    // larger edge lengths do not give significant benefits in AABB-tree optimized searches
    const float approxMaxEdgeLen = 5 * voxelSize;

    // subdivide copied mesh
    SubdivideSettings subSettings
    {
        .maxEdgeLen = approxMaxEdgeLen,
        .maxEdgeSplits = int( subMesh.area() / sqr( approxMaxEdgeLen ) ),
        .maxDeviationAfterFlip = 0.1f * voxelSize,
        .progressCallback = subprogress( cb, 0.25f, 0.75f )
    };
    subdivideMesh( subMesh, subSettings );
    if ( !reportProgress( cb, 0.75f ) )
        return unexpectedOperationCanceled();

    // pack subdivided mesh
    subMesh.packOptimally();
    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return subMesh;
}

int subdivideMesh( ObjectMeshData & data, const SubdivideSettings & settings )
{
    MR_TIMER;
    if ( !data.mesh )
    {
        assert( false );
        return 0;
    }

    auto notFlippable = data.selectedEdges | data.creases;

    MeshAttributesToUpdate meshParams;
    if ( !data.uvCoordinates.empty() )
        meshParams.uvCoords = &data.uvCoordinates;
    if ( !data.vertColors.empty() )
        meshParams.colorMap = &data.vertColors;
    if ( !data.texturePerFace.empty() )
        meshParams.texturePerFace = &data.texturePerFace;
    if ( !data.faceColors.empty() )
        meshParams.faceColors = &data.faceColors;

    auto updateAttributesCb = meshOnEdgeSplitAttribute( *data.mesh, meshParams );

    auto subs1 = settings;
    if ( data.selectedFaces.any() )
    {
        if ( subs1.region )
        {
            // given region must not include any face not from current face selection
            assert( subs1.region->is_subset_of( data.selectedFaces ) );
            assert( !subs1.maintainRegion );
            subs1.maintainRegion = &data.selectedFaces;
        }
        else // set face selection as subdivision region
            subs1.region = &data.selectedFaces;
    }

    subs1.onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
    {
        if ( data.selectedEdges.test( e.undirected() ) )
        {
            data.selectedEdges.autoResizeSet( e1.undirected() );
            notFlippable.autoResizeSet( e1.undirected() );
        }
        if ( data.creases.test( e.undirected() ) )
        {
            data.creases.autoResizeSet( e1.undirected() );
            notFlippable.autoResizeSet( e1.undirected() );
        }

        updateAttributesCb( e1, e );
        if ( settings.onEdgeSplit )
            settings.onEdgeSplit( e1, e );
    };
    assert( !subs1.notFlippable );
    subs1.notFlippable = &notFlippable;

    return subdivideMesh( *data.mesh, subs1 );
}

ObjectMeshData makeSubdividedObjectMeshData( const ObjectMesh & obj, const SubdivideSettings& settings )
{
    MR_TIMER;

    ObjectMeshData data = obj.data();
    if ( !data.mesh )
    {
        assert( false );
        return data;
    }
    // clone mesh as well
    data.mesh = std::make_shared<Mesh>( *data.mesh );
    subdivideMesh( data, settings );
    return data;
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
