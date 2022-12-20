#include "MRMeshDecimate.h"
#include "MRAABBTree.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

DecimateResult decimateParallelMeshInplace( MR::Mesh & mesh, const DecimateParallelInplaceSettings & settings )
{
    MR_TIMER

    DecimateResult res;
    if ( settings.subdivideParts <= 1 )
    {
        res = decimateMesh( mesh, settings );
        return res;
    }

    // not implemented in parallel version of decimate:
    assert( settings.maxDeletedVertices == INT_MAX );
    assert( settings.maxDeletedFaces == INT_MAX );
    assert( !settings.bdVerts );
    assert( !settings.vertForms );

    MR_WRITER( mesh );
    const auto & tree = mesh.getAABBTree();
    const auto subroots = tree.getSubtrees( settings.subdivideParts );
    const auto sz = subroots.size();
    if ( settings.progressCallback && !settings.progressCallback( 0.05f ) )
        return res;

    struct alignas(64) Parts
    {
        FaceBitSet faces;
        VertBitSet bdVerts;
        DecimateResult decimRes;
    };
    std::vector<Parts> parts( sz );

    // determine faces for each part
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ), [&]( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            parts[i].faces = tree.getSubtreeFaces( subroots[i] );
            parts[i].bdVerts = getBoundaryVerts( mesh.topology, &parts[i].faces );
        }
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.1f ) )
        return res;

    // determine edges in between the parts
    UndirectedEdgeBitSet stableEdges( mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( stableEdges, [&]( UndirectedEdgeId ue )
    {
        FaceId l = mesh.topology.left( ue );
        FaceId r = mesh.topology.right( ue );
        if ( !l || !r )
            return;
        for ( size_t i = 0; i < sz; ++i )
        {
            if ( parts[i].faces.test( l ) != parts[i].faces.test( r ) )
            {
                stableEdges.set( ue );
                break;
            }
        }
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.14f ) )
        return res;

    mesh.topology.preferEdges( stableEdges );
    if ( settings.progressCallback && !settings.progressCallback( 0.16f ) )
        return res;

    // compute quadratic form in each vertex
    auto mVertForms = computeFormsAtVertices( MeshPart{ mesh, settings.region }, settings.stabilizer );
    if ( settings.progressCallback && !settings.progressCallback( 0.2f ) )
        return res;

    mesh.topology.stopUpdatingValids();
    const auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> cancelled{ false };
    std::atomic<int> finishedParts{ 0 };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ), [&]( const tbb::blocked_range<size_t>& range )
    {
        const bool reportProgressFromThisThread = settings.progressCallback && mainThreadId == std::this_thread::get_id();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto reportThreadProgress = [&]( float p )
            {
                if ( cancelled.load( std::memory_order_relaxed ) )
                    return false;
                if ( reportProgressFromThisThread && !settings.progressCallback( 0.2f + 0.65f * ( finishedParts.load( std::memory_order_relaxed ) + p ) / sz ) )
                {
                    cancelled.store( true, std::memory_order_relaxed );
                    return false;
                }
                return true;
            };
            if ( !reportThreadProgress( 0 ) )
                break;

            DecimateSettings subSeqSettings = settings;
            subSeqSettings.vertForms = &mVertForms;
            subSeqSettings.touchBdVertices = false;
            subSeqSettings.bdVerts = &parts[i].bdVerts;
            FaceBitSet myRegion = parts[i].faces;
            if ( settings.region )
                myRegion &= *settings.region;
            subSeqSettings.region = &myRegion;
            if ( reportProgressFromThisThread )
                subSeqSettings.progressCallback = [reportThreadProgress]( float p ) { return reportThreadProgress( p ); };
            else if ( settings.progressCallback )
                subSeqSettings.progressCallback = [&cancelled]( float ) { return !cancelled.load( std::memory_order_relaxed ); };
            parts[i].decimRes = decimateMesh( mesh, subSeqSettings );

            if ( parts[i].decimRes.cancelled || !reportThreadProgress( 1 ) )
                break;
            finishedParts.fetch_add( 1, std::memory_order_relaxed );
        }
    } );

    if ( cancelled.load( std::memory_order_relaxed ) || ( settings.progressCallback && !settings.progressCallback( 0.85f ) ) )
        return res;

    mesh.topology.computeValidsFromEdges();

    if ( settings.progressCallback && !settings.progressCallback( 0.9f ) )
        return res;

    DecimateSettings seqSettings = settings;
    if ( settings.progressCallback )
        seqSettings.progressCallback = [cb = settings.progressCallback](float p) { return cb( 0.9f + 0.1f * p ); };
    res = decimateMesh( mesh, seqSettings );
    // update res from submesh decimations
    for ( const auto & submesh : parts )
    {
        res.facesDeleted += submesh.decimRes.facesDeleted;
        res.vertsDeleted += submesh.decimRes.vertsDeleted;
    }

    return res;
}

} //namespace MR
