#include "MRMeshTopology.h"
#include "MRBitSet.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRingIterator.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

MeshTopology::MeshTopology(MR::MeshTopology* mtop)
: mtop_( mtop )
{}

MeshTopology::~MeshTopology()
{
    delete mtop_;
}

VertBitSetReadOnly^ MeshTopology::ValidVerts::get()
{
    if ( !validVerts_ )
        validVerts_ = gcnew VertBitSet( new MR::VertBitSet( mtop_->getValidVerts() ) );

    return validVerts_;
}

FaceBitSetReadOnly^ MeshTopology::ValidFaces::get()
{
    if ( !validFaces_ )
        validFaces_ = gcnew FaceBitSet( new MR::FaceBitSet( mtop_->getValidFaces() ) );

    return validFaces_;
}

TriangulationReadOnly^ MeshTopology::Triangulation::get()
{
    if ( !triangulation_ )
    {
        int size = int( mtop_->faceSize() );
        triangulation_ = gcnew MR::DotNet::Triangulation( size );
        for ( size_t i = 0; i < size; i++ )
        {
            MR::ThreeVertIds threeVerts;
            MR::EdgeId a = mtop_->edgeWithLeft( MR::FaceId( i ) );
            threeVerts[0] = mtop_->org( a );
            MR::EdgeId b = mtop_->prev( a.sym() );            
            threeVerts[1] = mtop_->org(b);
            MR::EdgeId c = mtop_->prev( b.sym() );
            threeVerts[2] = mtop_->org( c );

            triangulation_->Add( MR::DotNet::ThreeVertIds { threeVerts[0], threeVerts[1], threeVerts[2] } );
        }
    }

    return triangulation_->AsReadOnly();
}

bool MeshTopology::isLoneEdge_( MR::EdgeId a )
{
    const auto& edges = mtop_->edges();
    if ( a >= edges.size() )
        return true;
    auto& adata = edges[a];
    if ( adata.left.valid() || adata.org.valid() || adata.next != a || adata.prev != a )
        return false;

    auto b = a.sym();
    auto& bdata = edges[b];
    if ( bdata.left.valid() || bdata.org.valid() || bdata.next != b || bdata.prev != b )
        return false;

    return true;
}

EdgePathReadOnly^ MeshTopology::HoleRepresentiveEdges::get()
{
    if ( !holeRepresentiveEdges_ )
    {
        const int edgeSize = int( mtop_->edgeSize() );
        MR::EdgeBitSet bdEdges( edgeSize );
        MR::EdgeBitSet representativeEdges( edgeSize );

        for ( MR::EdgeId i = MR::EdgeId{ 0 }; i < edgeSize; i++ )
        {
            if ( !mtop_->left( i ) && !isLoneEdge_( i ) )
                bdEdges.set( i );
        }

        for ( MR::EdgeId e = MR::EdgeId{ 0 }; e < edgeSize; e++ )
        {
            if ( !bdEdges.test( e ) )
                continue;

            for ( MR::EdgeId ei : leftRing0(*mtop_, e) )
            {
                bdEdges.reset( ei );
            }

            holeRepresentiveEdges_->Add( EdgeId( e ) );
        }
    }
    return holeRepresentiveEdges_->AsReadOnly();
}

MR_DOTNET_NAMESPACE_END
