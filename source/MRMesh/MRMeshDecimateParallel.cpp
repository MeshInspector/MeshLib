#include "MRMeshDecimateParallel.h"
#include "MRMesh.h"
#include "MRAABBTree.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRQuadraticForm.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRTBB.h"

namespace MR
{

DecimateResult decimateParallelMesh( MR::Mesh & mesh, const DecimateParallelSettings & settings )
{
    MR_TIMER;

    DecimateSettings seqSettings;
    seqSettings.strategy = settings.strategy;
    seqSettings.maxError = settings.maxError;
    seqSettings.maxEdgeLen = settings.maxEdgeLen;
    seqSettings.maxTriangleAspectRatio = settings.maxTriangleAspectRatio;
    seqSettings.criticalTriAspectRatio = settings.criticalTriAspectRatio;
    seqSettings.stabilizer = settings.stabilizer;
    seqSettings.optimizeVertexPos = settings.optimizeVertexPos;
    seqSettings.region = settings.region;
    seqSettings.touchBdVertices = settings.touchBdVertices;
    seqSettings.maxAngleChange = settings.maxAngleChange;
    if ( settings.preCollapse )
    {
        seqSettings.preCollapse = [&mesh, cb = settings.preCollapse]( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos ) -> bool
        {
            return cb( mesh.topology.org( edgeToCollapse ), mesh.topology.dest( edgeToCollapse ), newEdgeOrgPos );
        };
    }
    if ( settings.adjustCollapse )
    {
        seqSettings.adjustCollapse = [&mesh, cb = settings.adjustCollapse]( MR::EdgeId edgeToCollapse, float & collapseErrorSq, Vector3f & collapsePos )
        {
            cb( mesh.topology.org( edgeToCollapse ), mesh.topology.dest( edgeToCollapse ), collapseErrorSq, collapsePos );
        };
    }

    DecimateResult res;
    if ( settings.subdivideParts <= 1 )
    {
        seqSettings.progressCallback = settings.progressCallback;
        res = decimateMesh( mesh, seqSettings );
        return res;
    }

    MR_WRITER( mesh );
    const auto & tree = mesh.getAABBTree();
    const auto subroots = tree.getSubtrees( settings.subdivideParts );
    const auto sz = subroots.size();

    if ( settings.progressCallback && !settings.progressCallback( 0.05f ) )
        return res;

    struct alignas(64) SubMesh
    {
        Mesh m;
        VertBitSet mBdVerts;
        MR::Vector<MR::QuadraticForm3f, MR::VertId> mVertForms;
        VertMap subVertToOriginal;
        FaceBitSet region;
        DecimateResult decimRes;
    };
    std::vector<SubMesh> submeshes( sz );

    const auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> cancelled{ false };
    std::atomic<int> finishedSubmeshes{ 0 };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        const bool reportProgressFromThisThread = settings.progressCallback && mainThreadId == std::this_thread::get_id();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto reportThreadProgress = [&]( float p )
            {
                if ( cancelled.load( std::memory_order_relaxed ) )
                    return false;
                if ( reportProgressFromThisThread && !settings.progressCallback( 0.05f + 0.7f * ( finishedSubmeshes.load( std::memory_order_relaxed ) + p ) / sz ) )
                {
                    cancelled.store( true, std::memory_order_relaxed );
                    return false;
                }
                return true;
            };
            if ( !reportThreadProgress( 0 ) )
                break;
            auto faces = tree.getSubtreeFaces( subroots[i] );
            auto & submesh = submeshes[i];
            VertMap vertSubToFull;
            FaceHashMap faceFullToSub;
            PartMapping map;
            map.tgt2srcVerts = &vertSubToFull;
            if ( settings.region )
                map.src2tgtFaces = &faceFullToSub;
            submesh.m.addPartByMask( mesh, faces, map );

            if ( !reportThreadProgress( 0.1f ) )
                break;

            auto subSeqSettings = seqSettings;
            subSeqSettings.touchBdVertices = false;
            subSeqSettings.vertForms = &submesh.mVertForms;
            if ( settings.region )
            {
                submesh.region = settings.region->getMapping( faceFullToSub );
                subSeqSettings.region = &submesh.region;
            }
            if ( settings.preCollapse )
            {
                subSeqSettings.preCollapse = [&submesh, &vertSubToFull, cb = settings.preCollapse]( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos ) -> bool
                {
                    return cb( 
                        vertSubToFull[ submesh.m.topology.org( edgeToCollapse ) ],
                        vertSubToFull[ submesh.m.topology.dest( edgeToCollapse ) ],
                        newEdgeOrgPos );
                };
            }
            if ( settings.adjustCollapse )
            {
                subSeqSettings.adjustCollapse = [&submesh, &vertSubToFull, cb = settings.adjustCollapse]( MR::EdgeId edgeToCollapse, float & collapseErrorSq, Vector3f & collapsePos )
                {
                    cb( 
                        vertSubToFull[ submesh.m.topology.org( edgeToCollapse ) ],
                        vertSubToFull[ submesh.m.topology.dest( edgeToCollapse ) ],
                        collapseErrorSq, collapsePos );
                };
            }
            if ( reportProgressFromThisThread )
                subSeqSettings.progressCallback = [reportThreadProgress]( float p ) { return reportThreadProgress( 0.1f + 0.75f * p ); };
            else if ( settings.progressCallback )
                subSeqSettings.progressCallback = [&cancelled]( float ) { return !cancelled.load( std::memory_order_relaxed ); };
            submesh.decimRes = decimateMesh( submesh.m, subSeqSettings );
            if ( submesh.decimRes.cancelled || !reportThreadProgress( 0.85f ) )
                break;

            VertMap vertSubToPacked;
            FaceMap faceSubToPacked;
            submesh.m.pack( settings.region ? &faceSubToPacked : nullptr, &vertSubToPacked );
            if ( settings.region )
                submesh.region = submesh.region.getMapping( faceSubToPacked );

            if ( !reportThreadProgress( 0.9f ) )
                break;

            submesh.subVertToOriginal.resize( submesh.m.topology.lastValidVert() + 1 );
            for ( VertId beforePackId{ 0 }; beforePackId < vertSubToPacked.size(); ++beforePackId )
            {
                VertId packedId = vertSubToPacked[beforePackId];
                if ( packedId )
                {
                    submesh.subVertToOriginal[packedId] = vertSubToFull[beforePackId];
                    assert( packedId <= beforePackId );
                    submesh.mVertForms[packedId] = submesh.mVertForms[beforePackId];
                }
            }
            submesh.mBdVerts = submesh.m.topology.findBoundaryVerts();
            finishedSubmeshes.fetch_add( 1, std::memory_order_relaxed );
        }
    } );

    if ( cancelled.load( std::memory_order_relaxed ) || ( settings.progressCallback && !settings.progressCallback( 0.75f ) ) )
        return res;

    // recombine mesh from parts
    MR::Vector<MR::QuadraticForm3f, MR::VertId> unitedVertForms( mesh.topology.vertSize() );
    VertBitSet bdOfSomePiece( mesh.topology.vertSize() );
    Triangulation t;
    if ( settings.region )
        settings.region->clear();
    for ( const auto & submesh : submeshes )
    {
        for ( auto f : submesh.m.topology.getValidFaces() )
        {
            ThreeVertIds tri;
            submesh.m.topology.getTriVerts( f, tri );
            for ( int i = 0; i < 3; ++i )
                tri[i] = submesh.subVertToOriginal[ tri[i] ];
            t.push_back( tri );
            if ( settings.region && submesh.region.test( f ) )
                settings.region->autoResizeSet( t.backId() );
        }
        for ( auto v : submesh.m.topology.getValidVerts() )
        {
            const auto fv = submesh.subVertToOriginal[v];
            if ( submesh.mBdVerts.test( v ) )
            {
                assert( mesh.points[fv] == submesh.m.points[v] );
                bdOfSomePiece.set( fv );
            }
            else
            {
                mesh.points[fv] = submesh.m.points[v];
                unitedVertForms[fv] = submesh.mVertForms[v];
            }
        }
    }

    if ( settings.progressCallback && !settings.progressCallback( 0.8f ) )
        return res;

    mesh.topology = MeshBuilder::fromTriangles( t );

    if ( settings.progressCallback && !settings.progressCallback( 0.85f ) )
        return res;

    BitSetParallelFor( bdOfSomePiece, [&]( VertId v )
    {
        unitedVertForms[v] = computeFormAtVertex( { mesh, settings.region }, v, settings.stabilizer );
    } );

    if ( settings.progressCallback && !settings.progressCallback( 0.9f ) )
        return res;

    seqSettings.vertForms = &unitedVertForms;
    if ( settings.progressCallback )
        seqSettings.progressCallback = [cb = settings.progressCallback](float p) { return cb( 0.9f + 0.1f * p ); };
    res = decimateMesh( mesh, seqSettings );
    // update res from submesh decimations
    for ( const auto & submesh : submeshes )
    {
        res.facesDeleted += submesh.decimRes.facesDeleted;
        res.vertsDeleted += submesh.decimRes.vertsDeleted;
    }

    return res;
}

} //namespace MR
