#include "MRComputeBoundingBox.h"
#include "MRBox.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRId.h"
#include "MRTimer.h"
#include "MRVector.h"
#include "MRPch/MRTBB.h"

namespace MR
{

class VertBoundingBoxCalc 
{
public:
    VertBoundingBoxCalc( const VertCoords & points, const VertBitSet & region, const AffineXf3f * toWorld ) 
        : points_( points ), region_( region ), toWorld_( toWorld ) { }
    VertBoundingBoxCalc( VertBoundingBoxCalc & x, tbb::split ) : points_( x.points_ ), region_( x.region_ ), toWorld_( x.toWorld_ ) { }
    void join( const VertBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box3f & box() const { return box_; }

    void operator()( const tbb::blocked_range<VertId> & r ) 
    {
        for ( VertId v = r.begin(); v < r.end(); ++v ) 
        {
            if ( region_.test( v ) )
                box_.include( toWorld_ ? (*toWorld_)( points_[v] ) : points_[v] );
        }
    }
            
private:
    const VertCoords & points_;
    const VertBitSet & region_;
    const AffineXf3f * toWorld_ = nullptr;
    Box3f box_;
};

Box3f computeBoundingBox( const VertCoords & points, const VertBitSet & region, const AffineXf3f * toWorld )
{
    MR_TIMER

    VertBoundingBoxCalc calc( points, region, toWorld );
    parallel_reduce( tbb::blocked_range<VertId>( 0_v, points.endId() ), calc );
    return calc.box();
}

} //namespace MR
