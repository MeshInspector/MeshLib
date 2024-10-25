#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRVectorTraits.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>

namespace MR
{

/// \defgroup BoxGroup Box
/// \ingroup MathGroup
/// \{

/// returns all corners of given box
template <typename T>
std::array<Vector3<T>, 8> getCorners( const Box<Vector3<T>> & box );

/// Box given by its min- and max- corners
template <typename V>
struct Box
{
public:
    using VTraits = VectorTraits<V>;
    using T = typename VTraits::BaseType;
    static constexpr int elements = VTraits::size;

    V min, max;

    /// min/max access by 0/1 index
    const V & operator []( int e ) const { return *( &min + e ); }
          V & operator []( int e )       { return *( &min + e ); }

    /// create invalid box by default
    Box() : min{ VTraits::diagonal( std::numeric_limits<T>::max() ) }, max{ VTraits::diagonal( std::numeric_limits<T>::lowest() ) } { }
    Box( const V& min, const V& max ) : min{ min }, max{ max } { }

    /// skip initialization of min/max
    template <typename VV = V, typename std::enable_if_t<VectorTraits<VV>::supportNoInit, int> = 0>
    explicit Box( NoInit ) : min{ noInit }, max{ noInit } { }

    template <typename VV = V, typename std::enable_if_t<!VectorTraits<VV>::supportNoInit, int> = 0>
    explicit Box( NoInit ) { }

    template <typename U>
    explicit Box( const Box<U> & a ) : min{ a.min }, max{ a.max } { }

    static Box fromMinAndSize( const V& min, const V& size ) { return Box{ min, V( min + size ) }; }

    /// true if the box contains at least one point
    bool valid() const
    {
        for ( int i = 0; i < elements; ++i )
            if ( VTraits::getElem( i, min ) > VTraits::getElem( i, max ) )
                return false;
        return true;
    }

    /// computes center of the box
    V center() const { assert( valid() ); return ( min + max ) / T(2); }

    /// computes size of the box in all dimensions
    V size() const { assert( valid() ); return max - min; }

    /// computes length from min to max
    T diagonal() const { return std::sqrt( sqr( size() ) ); }

    /// computes the volume of this box
    T volume() const
    {
        assert( valid() );
        T res{ 1 };
        for ( int i = 0; i < elements; ++i )
            res *= VTraits::getElem( i, max ) - VTraits::getElem( i, min );
        return res;
    }

    /// minimally increases the box to include given point
    void include( const V & pt )
    {
        for ( int i = 0; i < elements; ++i )
        {
            if ( VTraits::getElem( i, pt ) < VTraits::getElem( i, min ) ) VTraits::getElem( i, min ) = VTraits::getElem( i, pt );
            if ( VTraits::getElem( i, pt ) > VTraits::getElem( i, max ) ) VTraits::getElem( i, max ) = VTraits::getElem( i, pt );
        }
    }

    /// minimally increases the box to include another box
    void include( const Box & b )
    {
        for ( int i = 0; i < elements; ++i )
        {
            if ( VTraits::getElem( i, b.min ) < VTraits::getElem( i, min ) ) VTraits::getElem( i, min ) = VTraits::getElem( i, b.min );
            if ( VTraits::getElem( i, b.max ) > VTraits::getElem( i, max ) ) VTraits::getElem( i, max ) = VTraits::getElem( i, b.max );
        }
    }

    /// checks whether given point is inside (including the surface) of the box
    bool contains( const V & pt ) const
    {
        for ( int i = 0; i < elements; ++i )
            if ( VTraits::getElem( i, min ) > VTraits::getElem( i, pt ) || VTraits::getElem( i, pt ) > VTraits::getElem( i, max ) )
                return false;
        return true;
    }

    /// returns closest point in the box to given point
    V getBoxClosestPointTo( const V & pt ) const
    {
        assert( valid() );
        V res;
        for ( int i = 0; i < elements; ++i )
            VTraits::getElem( i, res ) = std::clamp( VTraits::getElem( i, pt ), VTraits::getElem( i, min ), VTraits::getElem( i, max ) );
        return res;
    }

    /// checks whether this box intersects or touches given box
    bool intersects( const Box & b ) const
    {
        for ( int i = 0; i < elements; ++i )
        {
            if ( VTraits::getElem( i, b.max ) < VTraits::getElem( i, min ) || VTraits::getElem( i, b.min ) > VTraits::getElem( i, max ) )
                return false;
        }
        return true;
    }

    /// computes intersection between this and other box
    Box intersection( const Box & b ) const
    {
        Box res;
        for ( int i = 0; i < elements; ++i )
        {
            VTraits::getElem( i, res.min ) = std::max( VTraits::getElem( i, min ), VTraits::getElem( i, b.min ) );
            VTraits::getElem( i, res.max ) = std::min( VTraits::getElem( i, max ), VTraits::getElem( i, b.max ) );
        }
        return res;
    }
    Box & intersect( const Box & b ) { return *this = intersection( b ); }

    /// returns squared distance between this box and given one;
    /// returns zero if the boxes touch or intersect
    T getDistanceSq( const Box & b ) const
    {
        auto ibox = intersection( b );
        T distSq = 0;
        for ( int i = 0; i < elements; ++i )
            if ( VTraits::getElem( i, ibox.min ) > VTraits::getElem( i, ibox.max ) )
                distSq += sqr( VTraits::getElem( i, ibox.min ) - VTraits::getElem( i, ibox.max ) );
        return distSq;
    }

    /// returns squared distance between this box and given point;
    /// returns zero if the point is inside or on the boundary of the box
    T getDistanceSq( const V & pt ) const
    {
        assert( valid() );
        T res{};
        for ( int i = 0; i < elements; ++i )
        {
            if ( VTraits::getElem( i, pt ) < VTraits::getElem( i, min ) )
                res += sqr( VTraits::getElem( i, pt ) - VTraits::getElem( i, min ) );
            else
            if ( VTraits::getElem( i, pt ) > VTraits::getElem( i, max ) )
                res += sqr( VTraits::getElem( i, pt ) - VTraits::getElem( i, max ) );
        }
        return res;
    }

    /// decreases min and increased max on given value
    Box expanded( const V & expansion ) const
    {
        assert( valid() );
        return Box( min - expansion, max + expansion );
    }

    /// decreases min and increases max to their closest representable value
    Box insignificantlyExpanded() const
    {
        assert( valid() );
        Box res;
        for ( int i = 0; i < elements; ++i )
        {
            VTraits::getElem( i, res.min ) = std::nextafter( VTraits::getElem( i, min ), std::numeric_limits<T>::lowest() );
            VTraits::getElem( i, res.max ) = std::nextafter( VTraits::getElem( i, max ), std::numeric_limits<T>::max() );
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

/// find the tightest box enclosing this one after transformation
template <typename V>
inline Box<V> transformed( const Box<V> & box, const AffineXf<V> & xf )
{
    if ( !box.valid() )
        return {};
    Box<V> res;
    for ( const auto & p : getCorners( box ) )
        res.include( xf( p ) );
    return res;
}
/// this version returns input box as is if pointer to transformation is null
template <typename V>
inline Box<V> transformed( const Box<V> & box, const AffineXf<V> * xf )
{
    return xf ? transformed( box, *xf ) : box;
}

/// returns size along x axis
template <typename V>
inline auto width( const Box<V>& box )
{
    return box.max.x - box.min.x;
}

/// returns size along y axis
template <typename V>
inline auto height( const Box<V>& box )
{
    return box.max.y - box.min.y;
}

/// returns size along z axis
template <typename V>
inline auto depth( const Box<V>& box )
{
    return box.max.z - box.min.z;
}

/// get<0> returns min, get<1> returns max
template<size_t I, typename V>
constexpr const V& get( const Box<V>& box ) noexcept { return box[int( I )]; }
template<size_t I, typename V>
constexpr       V& get(       Box<V>& box ) noexcept { return box[int( I )]; }

/// \}

} // namespace MR

namespace std
{

template<size_t I, typename V>
struct tuple_element<I, MR::Box<V>> { using type = V; };

template <typename V> 
struct tuple_size<MR::Box<V>> : integral_constant<size_t, 2> {};

} //namespace std
