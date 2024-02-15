#include "MRDenseBox.h"
#include "MRMesh.h"

namespace MR
{

DenseBox::DenseBox( const MeshPart& meshPart, const AffineXf3f* xf /*= nullptr*/ )
{
    include_( meshPart, xf );
}

DenseBox::DenseBox( const PointCloud& points, const AffineXf3f* xf /*= nullptr*/ )
{
    include_( points, xf );
}

DenseBox::DenseBox( const Polyline3& line, const AffineXf3f* xf /*= nullptr*/ )
{
    include_( line, xf );
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

void DenseBox::include_( const MeshPart& meshPart, const AffineXf3f* xf /*= nullptr */ )
{
    PointAccumulator accum;
    accumulateFaceCenters( accum, meshPart, xf );
    if ( !accum.valid() )
        return;
    basisXf_ = AffineXf3f( accum.getBasicXf() );
    basisXfInv_ = basisXf_.inverse();
    auto tempXf = basisXfInv_;
    if ( xf )
        tempXf = basisXfInv_ * ( *xf );

    box_.include( meshPart.mesh.computeBoundingBox( &tempXf ) );
}

void DenseBox::include_( const PointCloud& points, const AffineXf3f* xf )
{
    PointAccumulator accum;
    accumulatePoints( accum, points, xf );
    if ( !accum.valid() )
        return;
    basisXf_ = AffineXf3f( accum.getBasicXf() );
    basisXfInv_ = basisXf_.inverse();
    auto tempXf = basisXfInv_;
    if ( xf )
        tempXf = basisXfInv_ * ( *xf );

    box_.include( points.computeBoundingBox( &tempXf ) );
}

void DenseBox::include_( const Polyline3& line, const AffineXf3f* xf )
{
    PointAccumulator accum;
    accumulateLineCenters( accum, line, xf );
    if ( !accum.valid() )
        return;
    basisXf_ = AffineXf3f( accum.getBasicXf() );
    basisXfInv_ = basisXf_.inverse();
    auto tempXf = basisXfInv_;
    if ( xf )
        tempXf = basisXfInv_ * ( *xf );

    box_.include( line.computeBoundingBox( &tempXf ) );
}

}
