#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace MR
{

// returns all corners of given box
template <typename T>
std::array<Vector3<T>, 8> getCorners( const Box<Vector3<T>> & box );

// Box given by its min- and max- corners
template <typename V> 
struct Box
{
public:
    using T = typename V::ValueType;

    // create invalid box by default
    V min = V::diagonal( std::numeric_limits<T>::max() );
    V max = V::diagonal( std::numeric_limits<T>::lowest() );

    // min/max access by 0/1 index
    const V & operator []( int e ) const { return *( &min + e ); }
          V & operator []( int e )       { return *( &min + e ); }

    Box() = default;
    Box( const V& min, const V& max ) : min{ min }, max{ max } { }

    template <typename U>
    explicit Box( const Box<U> & a ) : min{ a.min }, max{ a.max } { }

    static Box fromMinAndSize( const V& min, const V size ) { return Box{ min,min + size }; }

    // true if the box contains at least one point
    bool valid() const 
    { 
        for ( int i = 0; i < V::elements; ++i )
            if ( min[i] > max[i] )
                return false;
        return true;
    }

    // computes center of the box
    V center() const { assert( valid() ); return ( min + max ) / T(2); }

    // computes size of the box in all dimensions
    V size() const { assert( valid() ); return max - min; }

    // computes length from min to max
    T diagonal() const { return size().length(); }

    // computes the volume of this box
    T volume() const
    {
        assert( valid() );
        T res{ 1 };
        for ( int i = 0; i < V::elements; ++i )
            res *= max[i] - min[i];
        return res;
    }

    // minimally increases the box to include given point
    void include( const V & pt )
    {
        for ( int i = 0; i < V::elements; ++i )
        {
            if ( pt[i] < min[i] ) min[i] = pt[i];
            if ( pt[i] > max[i] ) max[i] = pt[i];
        }
    }

    // minimally increases the box to include another box
    void include( const Box & b )
    {
        for ( int i = 0; i < V::elements; ++i )
        {
            if ( b.min[i] < min[i] ) min[i] = b.min[i];
            if ( b.max[i] > max[i] ) max[i] = b.max[i];
        }
    }

    // checks whether given point is inside (including the surface) of the box
    bool contains( const V & pt ) const
    {
        for ( int i = 0; i < V::elements; ++i )
            if ( min[i] > pt[i] || pt[i] > max[i] )
                return false;
        return true;
    }

    // returns closest point in the box to given point
    V getBoxClosestPointTo( const V & pt ) const
    {
        assert( valid() );
        V res;
        for ( int i = 0; i < V::elements; ++i )
            res[i] = std::clamp( pt[i], min[i], max[i] );
        return res;
    }

    // checks whether this box intersects or touches given box
    bool intersects( const Box & b ) const
    {
        for ( int i = 0; i < V::elements; ++i )
        {
            if ( b.max[i] < min[i] || b.min[i] > max[i] )
                return false;
        }
        return true;
    }

    // computes intersection between this and other box
    Box intersection( const Box & b ) const
    {
        Box res;
        for ( int i = 0; i < V::elements; ++i )
        {
            res.min[i] = std::max( min[i], b.min[i] );
            res.max[i] = std::min( max[i], b.max[i] );
        }
        return res;
    }
    Box & intersect( const Box & b ) { return *this = intersection( b ); }

    // returns squared distance between this box and given one;
    // returns zero if the boxes touch or intersect
    T getDistanceSq( const Box & b ) const
    {
        auto ibox = intersection( b );
        T distSq = 0;
        for ( int i = 0; i < V::elements; ++i )
            if ( ibox.min[i] > ibox.max[i] )
                distSq += sqr( ibox.min[i] - ibox.max[i] );
        return distSq;
    }

    // expands min and max to their closest representable value 
    Box insignificantlyExpanded() const
    {
        assert( valid() );
        Box res;
        for ( int i = 0; i < V::elements; ++i )
        {
            res.min[i] = std::nextafter( min[i], std::numeric_limits<T>::lowest() );
            res.max[i] = std::nextafter( max[i], std::numeric_limits<T>::max() );
        }
        return res;
    }

    bool operator == ( const Box & a ) const
        { return min == a.min && max == a.max;  }
    bool operator != ( const Box & a ) const
        { return !( *this == a ); }
};

template <typename T>
inline std::array<Vector3<T>, 8> getCorners( const Box<Vector3<T>> & box )
{
    return  
    {
        Vector3<T>{ box.min.x, box.min.y, box.min.z },
        Vector3<T>{ box.max.x, box.min.y, box.min.z },
        Vector3<T>{ box.min.x, box.max.y, box.min.z },
        Vector3<T>{ box.max.x, box.max.y, box.min.z },
        Vector3<T>{ box.min.x, box.min.y, box.max.z },
        Vector3<T>{ box.max.x, box.min.y, box.max.z },
        Vector3<T>{ box.min.x, box.max.y, box.max.z },
        Vector3<T>{ box.max.x, box.max.y, box.max.z }
    };
}

template <typename T>
inline std::array<Vector2<T>, 4> getCorners( const Box<Vector2<T>> & box )
{
    return  
    {
        Vector2<T>{ box.min.x, box.min.y },
        Vector2<T>{ box.max.x, box.min.y },
        Vector2<T>{ box.min.x, box.max.y },
        Vector2<T>{ box.max.x, box.max.y }
    };
}

// find the tightest box enclosing this one after transformation
template <typename V>
inline Box<V> transformed( const Box<V> & box, const AffineXf<V> & xf )
{
    Box<V> res;
    for ( const auto & p : getCorners( box ) )
        res.include( xf( p ) );
    return res;
}
// this version returns input box as is if pointer to transformation is null
template <typename V>
inline Box<V> transformed( const Box<V> & box, const AffineXf<V> * xf )
{
    return xf ? transformed( box, *xf ) : box;
}

// returns size along x axis
template <typename V>
inline auto width( const Box<V>& box )
{
    return box.max.x - box.min.x;
}

// returns size along y axis
template <typename V>
inline auto height( const Box<V>& box )
{
    return box.max.y - box.min.y;
}

// returns size along z axis
template <typename V>
inline auto depth( const Box<V>& box )
{
    return box.max.z - box.min.z;
}

} //namespace MR
