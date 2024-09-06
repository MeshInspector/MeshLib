#include "MRComputeBoundingBox.h"
#include "MRAffineXf.h"
#include "MRAffineXf2.h"
#include "MRBitSet.h"
#include "MRId.h"
#include "MRTimer.h"
#include "MRVector.h"
#include "MRPch/MRTBB.h"

namespace MR
{

template<typename V>
class VertBoundingBoxCalc 
{
public:
    VertBoundingBoxCalc( const Vector<V, VertId> & points, const VertBitSet * region, const AffineXf<V> * toWorld ) 
        : points_( points ), region_( region ), toWorld_( toWorld ) { }
    VertBoundingBoxCalc( VertBoundingBoxCalc & x, tbb::split ) : points_( x.points_ ), region_( x.region_ ), toWorld_( x.toWorld_ ) { }
    void join( const VertBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box<V> & box() const { return box_; }

    void operator()( const tbb::blocked_range<VertId> & r ) 
    {
        for ( VertId v = r.begin(); v < r.end(); ++v ) 
        {
            if ( !region_ || region_->test( v ) )
                box_.include( toWorld_ ? (*toWorld_)( points_[v] ) : points_[v] );
        }
    }
            
private:
    const Vector<V, VertId> & points_;
    const VertBitSet* region_ = nullptr;
    const AffineXf<V>* toWorld_ = nullptr;
    Box<V> box_;
};

template<typename V>
Box<V> computeBoundingBox( const Vector<V, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet * region, const AffineXf<V> * toWorld )
{
    MR_TIMER

    VertBoundingBoxCalc calc( points, region, toWorld );
    parallel_reduce( tbb::blocked_range<VertId>( firstVert, lastVert ), calc );
    return calc.box();
}

template MRMESH_API Box2f computeBoundingBox( const Vector<Vector2f, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet * region, const AffineXf2f * toWorld );
template MRMESH_API Box2d computeBoundingBox( const Vector<Vector2d, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet * region, const AffineXf2d * toWorld );
template MRMESH_API Box3f computeBoundingBox( const Vector<Vector3f, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet * region, const AffineXf3f * toWorld );
template MRMESH_API Box3d computeBoundingBox( const Vector<Vector3d, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet * region, const AffineXf3d * toWorld );

} //namespace MR
