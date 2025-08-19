#pragma once

#include "MRMacros.h"
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
    using Vb = typename VTraits::template ChangeBaseType<bool>; //if V is Vector3<T> then Vb is Vector3b
    using VbTraits = VectorTraits<Vb>;

    V min, max;

    /// min/max access by 0/1 index
    const V & operator []( int e ) const { return *( &min + e ); }
          V & operator []( int e )       { return *( &min + e ); }

    /// create invalid box by default
    Box() : min{ VTraits::diagonal( std::numeric_limits<T>::max() ) }, max{ VTraits::diagonal( std::numeric_limits<T>::lowest() ) } { }
    Box( const V& min, const V& max ) : min{ min }, max{ max } { }

    /// skip initialization of min/max
    #if MR_HAS_REQUIRES
    // If the compiler supports `requires`, use that instead of `std::enable_if` here.
    // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
    explicit Box( NoInit ) requires (VectorTraits<V>::supportNoInit) : min{ noInit }, max{ noInit } { }
    explicit Box( NoInit ) requires (!VectorTraits<V>::supportNoInit) { }
    #else
    template <typename VV = V, typename std::enable_if_t<VectorTraits<VV>::supportNoInit, int> = 0>
    explicit Box( NoInit ) : min{ noInit }, max{ noInit } { }
    template <typename VV = V, typename std::enable_if_t<!VectorTraits<VV>::supportNoInit, int> = 0>
    explicit Box( NoInit ) { }
    #endif

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

    /// returns the corner of this box as specified by given bool-vector:
    /// 0 element in (c) means take min's coordinate,
    /// 1 element in (c) means take max's coordinate
    V corner( const Vb& c ) const
    {
        V res;
        for ( int i = 0; i < elements; ++i )
            VTraits::getElem( i, res ) = VTraits::getElem( i, operator[]( VbTraits::getElem( i, c ) ) );
        return res;
    }

    /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
    /// finds the box's corner for which d is minimal
    static Vb getMinBoxCorner( const V & n )
    {
        Vb res;
        for ( int i = 0; i < elements; ++i )
            VbTraits::getElem( i, res ) = VTraits::getElem( i, n ) < 0;
        return res;
    }

    /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
    /// finds the box's corner for which d is maximal
    static Vb getMaxBoxCorner( const V & n )
    {
        return getMinBoxCorner( -n );
    }

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

    /// checks whether given point is inside (including the surface) of this box
    bool contains( const V & pt ) const
    {
        for ( int i = 0; i < elements; ++i )
            if ( VTraits::getElem( i, min ) > VTraits::getElem( i, pt ) || VTraits::getElem( i, pt ) > VTraits::getElem( i, max ) )
                return false;
        return true;
    }

    /// checks whether given box is fully inside (the surfaces may touch) of this box
    bool contains( const Box& otherbox ) const
    {
        for ( int i = 0; i < elements; ++i )
            if ( VTraits::getElem( i, min ) > VTraits::getElem( i, otherbox.min ) || VTraits::getElem( i, otherbox.max ) > VTraits::getElem( i, max ) )
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

    /// returns the closest point on the box to the given point
    /// for points outside the box this is equivalent to getBoxClosestPointTo
    V getProjection( const V & pt ) const
    {
        assert( valid() );
        if ( !contains( pt ) )
            return getBoxClosestPointTo( pt );

        T minDist = std::numeric_limits<T>::max();
        int minDistDim {};
        T minDistPos {};

        for ( auto dim = 0; dim < elements; ++dim )
        {
            for ( const auto& border : { min, max } )
            {
                if ( auto dist = std::abs( VTraits::getElem( dim, border ) - VTraits::getElem( dim, pt ) ); dist < minDist )
                {
                    minDist = dist;
                    minDistDim = dim;
                    minDistPos = VTraits::getElem( dim, border );
                }
            }
        }

        auto proj = pt;
        VTraits::getElem( minDistDim, proj ) = minDistPos;
        return proj;
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
    assert( box.valid() );
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
    assert( box.valid() );
    return
    {
        Vector2<T>{ box.min.x, box.min.y },
        Vector2<T>{ box.max.x, box.min.y },
        Vector2<T>{ box.min.x, box.max.y },
        Vector2<T>{ box.max.x, box.max.y }
    };
}

/// among all planes with given normal and arbitrary shift: dot(n,x) = d
/// finds minimal and maximal touching planes for given box, and returns them as { min_d, max_d }
template <typename V>
MinMax<typename Box<V>::T> getTouchPlanes( const Box<V> & box, const V & n )
{
    auto minCorner = box.getMinBoxCorner( n );
    auto maxCorner = minCorner;
    for ( auto & v : maxCorner )
        v = !v;
    return
    {
        dot( n, box.corner( minCorner ) ),
        dot( n, box.corner( maxCorner ) )
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

/// returns a vector with unique integer values, each representing a dimension of the box;
/// the dimensions are sorted according to the sizes of the box along each dimension:
/// the dimension with the smallest size is first, largest size - last.
/// E.g. findSortedBoxDims( Box3f( {0, 0, 0}, {2, 1, 3} ) ) == Vector3i(1, 0, 2)
template <typename V>
inline auto findSortedBoxDims( const Box<V>& box ) -> typename VectorTraits<V>::template ChangeBaseType<int>
{
    constexpr auto es = Box<V>::elements;
    auto boxDiag = box.max - box.min;
    std::pair<float, int> ps[es];
    for ( int i = 0; i < es; ++i )
        ps[i] = { boxDiag[i], i };

    // bubble sort (optimal for small array)
    for ( int i = 0; i + 1 < es; ++i )
        for ( int j = i + 1; j < es; ++j )
            if ( ps[j] < ps[i] )
                std::swap( ps[i], ps[j] );

    typename VectorTraits<V>::template ChangeBaseType<int> res( noInit );
    for ( int i = 0; i < es; ++i )
        res[i] = ps[i].second;
    return res;
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
