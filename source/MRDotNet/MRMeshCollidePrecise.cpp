#include "MRMeshCollidePrecise.h"
#include "MRBitSet.h"
#include "MRCoordinateConverters.h"
#include "MRAffineXf.h"

#pragma managed( push, off )
#include <MRMesh/MRBitSet.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

PreciseCollisionResult::PreciseCollisionResult( MR::PreciseCollisionResult* nativeResult )
{
    if ( !nativeResult )
        throw gcnew System::ArgumentNullException( "edgesAtrisB and edgesBtrisA cannot be null" );

    nativeResult_ = nativeResult;
}

PreciseCollisionResult::~PreciseCollisionResult()
{
    delete nativeResult_;
}

ReadOnlyCollection<EdgeTri>^ PreciseCollisionResult::EdgesAtrisB::get()
{
    if ( !edgesAtrisB_ )
    {
        edgesAtrisB_ = gcnew List<EdgeTri>( int( nativeResult_->edgesAtrisB.size() ) );
        for ( auto& nativeEdgeTri : nativeResult_->edgesAtrisB )
        {
            EdgeTri edgeTri;
            edgeTri.edge = EdgeId( nativeEdgeTri.edge );
            edgeTri.tri = FaceId( nativeEdgeTri.tri );
            edgesAtrisB_->Add( edgeTri );
        }
    }

    return edgesAtrisB_->AsReadOnly();
}

ReadOnlyCollection<EdgeTri>^ PreciseCollisionResult::EdgesBtrisA::get()
{
    if ( !edgesBtrisA_ )
    {
        edgesBtrisA_ = gcnew List<EdgeTri>( int( nativeResult_->edgesBtrisA.size() ) );
        for ( auto& nativeEdgeTri : nativeResult_->edgesBtrisA )
        {
            EdgeTri edgeTri;
            edgeTri.edge = EdgeId( nativeEdgeTri.edge );
            edgeTri.tri = FaceId( nativeEdgeTri.tri );
            edgesBtrisA_->Add( edgeTri );
        }
    }

    return edgesBtrisA_->AsReadOnly();
}

PreciseCollisionResult^ MeshCollidePrecise::FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv )
{
    return FindCollidingEdgeTrisPrecise( meshA, meshB, conv, nullptr, false );
}

PreciseCollisionResult^ MeshCollidePrecise::FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigibB2A )
{
    return FindCollidingEdgeTrisPrecise( meshA, meshB, conv, rigibB2A, false );
}

PreciseCollisionResult^ MeshCollidePrecise::FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigidB2A, bool anyInterssection )
{
    if ( !meshA.mesh )
        throw gcnew System::ArgumentNullException( "meshA" );
    if ( !meshB.mesh )
        throw gcnew System::ArgumentNullException( "meshB" );
    if ( !conv )
        throw gcnew System::ArgumentNullException( "conv" );

    MR::FaceBitSet nativeRegionA;
    MR::FaceBitSet nativeRegionB;
    if ( meshA.region )
        nativeRegionA = MR::FaceBitSet( meshA.region->bitSet()->m_bits.begin(), meshA.region->bitSet()->m_bits.end() );
    if ( meshB.region )
        nativeRegionB = MR::FaceBitSet( meshB.region->bitSet()->m_bits.begin(), meshB.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMeshA( *meshA.mesh->getMesh(), meshA.region ? &nativeRegionA : nullptr );
    MR::MeshPart nativeMeshB( *meshB.mesh->getMesh(), meshB.region ? &nativeRegionB : nullptr );

    auto nativeResult = MR::findCollidingEdgeTrisPrecise( nativeMeshA, nativeMeshB, *conv->getConvertToIntVector(), rigidB2A ? rigidB2A->xf() : nullptr, anyInterssection );
    return gcnew PreciseCollisionResult( new MR::PreciseCollisionResult( std::move( nativeResult ) ) );
}

MR_DOTNET_NAMESPACE_END
