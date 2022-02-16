#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

Box3f PointCloud::getBoundingBox() const 
{ 
    return getAABBTree().getBoundingBox(); 
}

class PointBoundingBoxCalc 
{
public:
    PointBoundingBoxCalc( const PointCloud & points, const AffineXf3f * toWorld ) : points_( points ), toWorld_( toWorld ) { }
    PointBoundingBoxCalc( PointBoundingBoxCalc & x, tbb::split ) : points_( x.points_ ), toWorld_( x.toWorld_ ) { }
    void join( const PointBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box3f & box() const { return box_; }

    void operator()( const tbb::blocked_range<VertId> & r ) 
    {
        for ( VertId v = r.begin(); v < r.end(); ++v ) 
        {
            if ( points_.validPoints.test( v ) )
                box_.include( toWorld_ ? (*toWorld_)( points_.points[v] ) : points_.points[v] );
        }
    }
            
private:
    const PointCloud & points_;
    const AffineXf3f * toWorld_ = nullptr;
    Box3f box_;
};

Box3f PointCloud::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    MR_TIMER

    PointBoundingBoxCalc calc( *this, toWorld );
    parallel_reduce( tbb::blocked_range<VertId>( 0_v, VertId( points.size() ) ), calc );
    return calc.box();
}

const AABBTreePoints& PointCloud::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePoints( *this ); } );
}

} //namespace MR

