#include "MREMeshDecimateParallel.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRQuadraticForm.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRPch/MRTBB.h"

namespace MRE
{

using namespace MR;

DecimateResult decimateParallelMesh( MR::Mesh & mesh, const DecimateParallelSettings & settings )
{
    MR_TIMER;

    DecimateSettings seqSettings;
    seqSettings.strategy = settings.strategy;
    seqSettings.maxError = settings.maxError;
    seqSettings.maxTriangleAspectRatio = settings.maxTriangleAspectRatio;
    seqSettings.stabilizer = settings.stabilizer;
    seqSettings.region = settings.region;
    if ( settings.preCollapse )
    {
        seqSettings.preCollapse = [&mesh, cb = settings.preCollapse]( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos ) -> bool
        {
            return cb( mesh.topology.org( edgeToCollapse ), mesh.topology.dest( edgeToCollapse ), newEdgeOrgPos );
        };
    }

    DecimateResult res;
    if ( settings.subdivideParts <= 1 )
    {
        seqSettings.progressCallback = settings.progressCallback;
        res = decimateMesh( mesh, seqSettings );
        return res;
    }

    MR_MESH_WRITER( mesh );
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
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        const bool reportProgressFromThisThread = settings.progressCallback && mainThreadId == std::this_thread::get_id();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto reportThreadProgress = [&]( float p )
            {
                if ( reportProgressFromThisThread && !settings.progressCallback( 0.05f + 0.7f * ( i - range.begin() + p ) / range.size() ) )
                {
                    cancelled.store( true, std::memory_order_relaxed );
                    return false;
                }
                return !cancelled.load( std::memory_order_relaxed );
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
        }
    } );

    if ( cancelled.load( std::memory_order_relaxed ) || settings.progressCallback && !settings.progressCallback( 0.75f ) )
        return res;

    // recombine mesh from parts
    MR::Vector<MR::QuadraticForm3f, MR::VertId> unitedVertForms( mesh.topology.vertSize() );
    VertBitSet bdOfSomePiece( mesh.topology.vertSize() );
    std::vector<MR::MeshBuilder::Triangle> tris;
    FaceId nextFace{ 0 };
    if ( settings.region )
        settings.region->clear();
    for ( const auto & submesh : submeshes )
    {
        for ( auto t : submesh.m.topology.getValidFaces() )
        {
            MR::MeshBuilder::Triangle tri;
            tri.f = nextFace;
            ++nextFace;
            submesh.m.topology.getTriVerts( t, tri.v );
            for ( int i = 0; i < 3; ++i )
                tri.v[i] = submesh.subVertToOriginal[ tri.v[i] ];
            tris.push_back( tri );
            if ( settings.region && submesh.region.test( t ) )
                settings.region->autoResizeSet( tri.f );
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

    mesh.topology = fromTriangles( tris );

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

} //namespace MRE
