#include "MRDenseBox.h"
#include "MRMesh.h"

namespace MR
{

DenseBox::DenseBox( const MeshPart& meshPart, const AffineXf3f* xf )
{
    include( meshPart, xf );
}

void DenseBox::include( const MeshPart& meshPart, const AffineXf3f* xf /*= nullptr */ )
{
    accumulateFaceCenters( accum_, meshPart, xf );
    basisXf_ = AffineXf3f( accum_.getBasicXf() );
    basisXfInv_ = basisXf_.inverse();
    auto tempXf = basisXfInv_;
    if ( xf )
        tempXf = basisXfInv_ * ( *xf );

    box_.include( meshPart.mesh.computeBoundingBox( meshPart.region, &tempXf ) );
}

Vector3f DenseBox::center() const
{
    return basisXf_( box_.center() );
}

Vector3f DenseBox::corner( const Vector3b& index ) const
{
    Vector3f res;
    for ( int i = 0; i < 3; ++i )
        res[i] = index[i] ? box_.max[i] : box_.min[i];
    return basisXf_( res );
}

bool DenseBox::contains( const Vector3f& pt ) const
{
    return box_.contains( basisXfInv_( pt ) );
}

}