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
    if ( settings.preCollapse )
    {
        seqSettings.preCollapse = [&mesh, cb = settings.preCollapse]( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos ) -> bool
        {
            return cb( mesh.topology.org( edgeToCollapse ), mesh.topology.dest( edgeToCollapse ), newEdgeOrgPos );
        };
    }

    if ( settings.subdivideParts <= 1 )
    {
        return decimateMesh( mesh, seqSettings );
    }

    MR_MESH_WRITER( mesh );
    const auto & tree = mesh.getAABBTree();
    const auto subroots = tree.getSubtrees( settings.subdivideParts );
    const auto sz = subroots.size();

    struct alignas(64) SubMesh
    {
        Mesh m;
        VertBitSet mBdVerts;
        MR::Vector<MR::QuadraticForm3f, MR::VertId> mVertForms;
        VertMap subVertToOriginal;
        DecimateResult decimRes;
    };
    std::vector<SubMesh> submeshes( sz );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto faces = tree.getSubtreeFaces( subroots[i] );
            auto & submesh = submeshes[i];
            VertMap subToFull;
            PartMapping map;
            map.tgt2srcVerts = &subToFull;
            submesh.m.addPartByMask( mesh, faces, map );

            auto subSeqSettings = seqSettings;
            subSeqSettings.touchBdVertices = false;
            subSeqSettings.vertForms = &submesh.mVertForms;
            if ( settings.preCollapse )
            {
                subSeqSettings.preCollapse = [&submesh, &subToFull, cb = settings.preCollapse]( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos ) -> bool
                {
                    return cb( 
                        subToFull[ submesh.m.topology.org( edgeToCollapse ) ],
                        subToFull[ submesh.m.topology.dest( edgeToCollapse ) ],
                        newEdgeOrgPos );
                };
            }
            submesh.decimRes = decimateMesh( submesh.m, subSeqSettings );

            VertMap subToPacked;
            submesh.m.pack( nullptr, &subToPacked );

            submesh.subVertToOriginal.resize( submesh.m.topology.lastValidVert() + 1 );
            for ( VertId beforePackId{ 0 }; beforePackId < subToPacked.size(); ++beforePackId )
            {
                VertId packedId = subToPacked[beforePackId];
                if ( packedId )
                {
                    submesh.subVertToOriginal[packedId] = subToFull[beforePackId];
                    assert( packedId <= beforePackId );
                    submesh.mVertForms[packedId] = submesh.mVertForms[beforePackId];
                }
            }
            submesh.mBdVerts = submesh.m.topology.findBoundaryVerts();
        }
    } );

    // recombine mesh from parts
    MR::Vector<MR::QuadraticForm3f, MR::VertId> unitedVertForms( mesh.topology.vertSize() );
    VertBitSet bdOfSomePiece( mesh.topology.vertSize() );
    std::vector<MR::MeshBuilder::Triangle> tris;
    FaceId nextFace{ 0 };
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
    mesh.topology = fromTriangles( tris );
    BitSetParallelFor( bdOfSomePiece, [&]( VertId v )
    {
        unitedVertForms[v] = computeFormAtVertex( mesh, v, settings.stabilizer );
    } );

    seqSettings.vertForms = &unitedVertForms;
    auto res = decimateMesh( mesh, seqSettings );
    // update res from submesh decimations
    for ( const auto & submesh : submeshes )
    {
        res.facesDeleted += submesh.decimRes.facesDeleted;
        res.vertsDeleted += submesh.decimRes.vertsDeleted;
    }

    return res;
}

} //namespace MRE
