#include "MRDenseBox.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

DenseBox::DenseBox( const std::vector<Vector3f>& points, const AffineXf3f* xf )
{
    init_( points, nullptr, xf );
}

DenseBox::DenseBox( const std::vector<Vector3f>& points, const std::vector<float>& weights, const AffineXf3f* xf )
{
    init_( points, &weights, xf );
}

DenseBox::DenseBox( const MeshPart& meshPart, const AffineXf3f* xf /*= nullptr*/ )
{
    init_( meshPart, xf );
}

DenseBox::DenseBox( const PointCloud& points, const AffineXf3f* xf /*= nullptr*/ )
{
    init_( points, xf );
}

DenseBox::DenseBox( const Polyline3& line, const AffineXf3f* xf /*= nullptr*/ )
{
    init_( line, xf );
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

void DenseBox::init_( const std::vector<Vector3f>& points, const std::vector<float>* weights, const AffineXf3f* xf )
{
    MR_TIMER
    for ( const auto& p : points )
        box_.include( xf ? (*xf)( p ) : p );
    if ( xf )
    {
        basisXf_ = *xf;
        basisXfInv_ = basisXf_.inverse();
    }

    assert( !weights || points.size() == weights->size() );
    PointAccumulator accum;
    if ( weights )
        accumulateWeighedPoints( accum, points, *weights, xf );
    else
        accumulatePoints( accum, points, xf );
    if ( !accum.valid() )
        return;

    const auto aXf = AffineXf3f( accum.getBasicXf() );
    const auto aInvXf = aXf.inverse();
    const auto tempXf = xf ? aInvXf * ( *xf ) : aInvXf;
    Box3f abox;
    for ( const auto& p : points )
        abox.include( tempXf( p ) );
    if ( abox.volume() < box_.volume() )
    {
        box_ = abox;
        basisXf_ = aXf;
        basisXfInv_ = aInvXf;
    }
}

void DenseBox::init_( const MeshPart& meshPart, const AffineXf3f* xf )
{
    MR_TIMER
    box_ = meshPart.mesh.computeBoundingBox( meshPart.region, xf );
    if ( xf )
    {
        basisXf_ = *xf;
        basisXfInv_ = basisXf_.inverse();
    }

    PointAccumulator accum;
    accumulateFaceCenters( accum, meshPart, xf );
    if ( !accum.valid() )
        return;

    const auto aXf = AffineXf3f( accum.getBasicXf() );
    const auto aInvXf = aXf.inverse();
    const auto tempXf = xf ? aInvXf * ( *xf ) : aInvXf;
    const auto abox = meshPart.mesh.computeBoundingBox( meshPart.region, &tempXf );
    if ( abox.volume() < box_.volume() )
    {
        box_ = abox;
        basisXf_ = aXf;
        basisXfInv_ = aInvXf;
    }
}

void DenseBox::init_( const PointCloud& points, const AffineXf3f* xf )
{
    MR_TIMER
    box_ = points.computeBoundingBox( xf );
    if ( xf )
    {
        basisXf_ = *xf;
        basisXfInv_ = basisXf_.inverse();
    }

    PointAccumulator accum;
    accumulatePoints( accum, points, xf );
    if ( !accum.valid() )
        return;

    const auto aXf = AffineXf3f( accum.getBasicXf() );
    const auto aInvXf = aXf.inverse();
    const auto tempXf = xf ? aInvXf * ( *xf ) : aInvXf;
    const auto abox = points.computeBoundingBox( &tempXf );
    if ( abox.volume() < box_.volume() )
    {
        box_ = abox;
        basisXf_ = aXf;
        basisXfInv_ = aInvXf;
    }
}

void DenseBox::init_( const Polyline3& line, const AffineXf3f* xf )
{
    MR_TIMER
    box_ = line.computeBoundingBox( xf );
    if ( xf )
    {
        basisXf_ = *xf;
        basisXfInv_ = basisXf_.inverse();
    }

    PointAccumulator accum;
    accumulateLineCenters( accum, line, xf );
    if ( !accum.valid() )
        return;

    const auto aXf = AffineXf3f( accum.getBasicXf() );
    const auto aInvXf = aXf.inverse();
    const auto tempXf = xf ? aInvXf * ( *xf ) : aInvXf;
    const auto abox = line.computeBoundingBox( &tempXf );
    if ( abox.volume() < box_.volume() )
    {
        box_ = abox;
        basisXf_ = aXf;
        basisXfInv_ = aInvXf;
    }
}

} //namespace MR
