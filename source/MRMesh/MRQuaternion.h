#pragma once

#include "MRAffineXf3.h"

namespace MR
{

/// Represents a quaternion following the notations from
/// https://en.wikipedia.org/wiki/Quaternion
/// \ingroup MathGroup
template <typename T>
struct Quaternion
{
    T a = 1; ///< real part of the quaternion
    T b = 0, c = 0, d = 0; ///< imaginary part: b*i + c*j + d*k

    constexpr Quaternion() noexcept = default;
    constexpr Quaternion( T a, T b, T c, T d ) noexcept : a( a ), b( b ), c( c ), d( d ) { }
    constexpr Quaternion( const Vector3<T> & axis, T angle ) noexcept;
    constexpr Quaternion( T real, const Vector3<T> & im ) noexcept : a( real ), b( im.x ), c( im.y ), d( im.z ) { }
    constexpr Quaternion( const Matrix3<T> & m );
    /// finds shorter arc rotation quaternion from one vector to another
    constexpr Quaternion( const Vector3<T>& from, const Vector3<T>& to ) noexcept;

    /// returns imaginary part of the quaternion as a vector
    [[nodiscard]] constexpr Vector3<T> im() const noexcept { return Vector3<T>{ b, c, d }; }

    /// returns angle of rotation encoded in this quaternion
    [[nodiscard]] constexpr T angle() const noexcept { return 2 * std::acos( std::clamp( a, T(-1), T(1) ) ); }
    /// returns axis of rotation encoded in this quaternion
    [[nodiscard]] constexpr Vector3<T> axis() const noexcept { return im().normalized(); }

    [[nodiscard]] constexpr T normSq() const { return a * a + b * b + c * c + d * d; }
    [[nodiscard]] constexpr T norm() const { return std::sqrt( normSq() ); }
    /// returns quaternion representing the same rotation, using the opposite rotation direction and opposite angle
    [[nodiscard]] constexpr Quaternion operator-() const { return {-a, -b, -c, -d}; }

    /// scales this quaternion to make its norm unit
    void normalize() { if ( T n = norm(); n > 0 ) *this /= n; }
    [[nodiscard]] Quaternion normalized() const { Quaternion res( *this ); res.normalize(); return res; }

    /// computes conjugate quaternion, which for unit quaternions encodes the opposite rotation
    [[nodiscard]] constexpr Quaternion conjugate() const noexcept { return {a, -b, -c, -d}; }
    /// computes reciprocal quaternion
    [[nodiscard]] constexpr Quaternion inverse() const noexcept { return conjugate() / normSq(); }
    /// for unit quaternion returns the rotation of point p, which is faster to compute for single point;
    /// for multiple points it is faster to create matrix representation and apply it to the points
    [[nodiscard]] constexpr Vector3<T> operator()( const Vector3<T> & p ) const noexcept;

    /// converts this into 3x3 rotation matrix
    [[nodiscard]] operator Matrix3<T>() const;

    /// given t in [0,1], interpolates linearly two quaternions giving in general not-unit quaternion
    [[nodiscard]] static Quaternion lerp( const Quaternion & q0, const Quaternion & q1, T t ) { return ( 1 - t ) * q0 + t * q1; }
    /// given t in [0,1] and two unit quaternions, interpolates them spherically and produces another unit quaternion
    [[nodiscard]] static Quaternion slerp( Quaternion q0, Quaternion q1, T t );
    /// given t in [0,1] and two rotation matrices, interpolates them spherically and produces another rotation matrix
    [[nodiscard]] static Matrix3<T> slerp( const Matrix3<T> & m0, const Matrix3<T> & m1, T t ) { return slerp( Quaternion<T>{ m0 }, Quaternion<T>{ m1 }, t ); }
    /// given t in [0,1] and rigid transformations, interpolates them spherically and produces another rigid transformation;
    /// p is the only point that will have straight line movement during interpolation
    [[nodiscard]] static AffineXf3<T> slerp( const AffineXf3<T> & xf0, const AffineXf3<T> & xf1, T t, const Vector3<T> & p = {} )
    {
        auto xfA = slerp( xf0.A, xf1.A, t );
        return { xfA, ( 1 - t ) * xf0( p ) + t * xf1( p ) - xfA * p };
    }

    Quaternion & operator *=( T s ) { a *= s; b *= s; c *= s; d *= s; return * this; }
    Quaternion & operator /=( T s ) { return *this *= ( 1 / s ); }
};

/// \related Quaternion
/// \{

template <typename T>
constexpr Quaternion<T>::Quaternion( const Vector3<T> & axis, T angle ) noexcept
{
    a = std::cos( angle / 2 );
    Vector3<T> im = std::sin( angle / 2 ) * axis.normalized();
    b = im.x;
    c = im.y;
    d = im.z;
}

template <typename T>
constexpr Quaternion<T>::Quaternion( const Matrix3<T> & m )
{
    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    const auto tr = m.trace();
    if ( tr > 0 )
    {
        auto S = std::sqrt( tr + 1 ) * 2;
        a = T( 0.25 ) * S;
        b = ( m.z.y - m.y.z ) / S;
        c = ( m.x.z - m.z.x ) / S;
        d = ( m.y.x - m.x.y ) / S;
    }
    else if ( m.x.x > m.y.y && m.x.x > m.z.z )
    {
        auto S = std::sqrt( 1 + m.x.x - m.y.y - m.z.z ) * 2;
        a = ( m.z.y - m.y.z ) / S;
        b = T( 0.25 ) * S;
        c = ( m.x.y + m.y.x ) / S;
        d = ( m.x.z + m.z.x ) / S;
    }
    else if ( m.y.y > m.z.z )
    {
        auto S = std::sqrt( 1 + m.y.y - m.x.x - m.z.z ) * 2;
        a = ( m.x.z - m.z.x ) / S;
        b = ( m.x.y + m.y.x ) / S;
        c = T( 0.25 ) * S;
        d = ( m.y.z + m.z.y ) / S;
    }
    else
    {
        auto S = std::sqrt( 1 + m.z.z - m.x.x - m.y.y ) * 2;
        a = ( m.y.x - m.x.y ) / S;
        b = ( m.x.z + m.z.x ) / S;
        c = ( m.y.z + m.z.y ) / S;
        d = T( 0.25 ) * S;
    }
}

template <typename T>
constexpr Quaternion<T>::Quaternion( const Vector3<T>& from, const Vector3<T>& to) noexcept
{
    // https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    a = dot( from, to );
    auto cr = cross( from, to );
    if( cr.x == 0 && cr.y == 0 && cr.z == 0 )
    {
        if( a >= 0 )
        {
            // parallel co-directional vectors
            // zero rotation
            a = 1; b = 0; c = 0; d = 0;
        }
        else
        {
            // parallel opposing vectors
            // any perpendicular axis Pi rotation
            auto perp = cross( from, from.furthestBasisVector() );
            a = 0; b = perp.x; c = perp.y; d = perp.z;
            normalize();
        }
    }
    else
    {
        a += std::sqrt( from.lengthSq() * to.lengthSq() );
        b = cr.x; c = cr.y; d = cr.z;
        normalize();
    }
}

template <typename T>
Quaternion<T> Quaternion<T>::slerp( Quaternion q0, Quaternion q1, T t )
{
    // https://en.wikipedia.org/wiki/Slerp
    q0.normalize();
    q1.normalize();

    T cosTheta = std::clamp( dot( q0, q1 ), T(-1), T(1) );
    if ( cosTheta < 0 )
    {
        q0 = -q0;
        cosTheta = -cosTheta;
    }
    T theta = std::acos( cosTheta );
    T sinTheta = std::sin( theta );
    if ( sinTheta <= 0 )
        return lerp( q0, q1, t ).normalized();

    return std::sin( theta * ( 1 - t ) ) / sinTheta * q0 + std::sin( theta * t ) / sinTheta * q1;
}

template <typename T>
Quaternion<T>::operator Matrix3<T>() const
{
    Matrix3<T> res;
    res.x = Vector3<T>{ a * a + b * b - c * c - d * d,      2*(b * c - a * d),             2*(b * d + a * c)        };
    res.y = Vector3<T>{      2*(b * c + a * d),        a * a + c * c - b * b - d * d,      2*(c * d - a * b)        };
    res.z = Vector3<T>{      2*(b * d - a * c),             2*(c * d + a * b),        a * a + d * d - b * b - c * c };
    return res;
}

template <typename T>
[[nodiscard]] inline bool operator ==( const Quaternion<T> & a, const Quaternion<T> & b )
{
    return a.a == b.a && a.b == b.b && a.c == b.c && a.d == b.d;
}

template <typename T>
[[nodiscard]] inline bool operator !=( const Quaternion<T> & a, const Quaternion<T> & b )
{
    return !( a == b );
}

template <typename T>
[[nodiscard]] inline Quaternion<T> operator +( const Quaternion<T> & a, const Quaternion<T> & b )
{
    return {a.a + b.a, a.b + b.b, a.c + b.c, a.d + b.d};
}

template <typename T>
[[nodiscard]] inline Quaternion<T> operator -( const Quaternion<T> & a, const Quaternion<T> & b )
{
    return {a.a - b.a, a.b - b.b, a.c - b.c, a.d - b.d};
}

template <typename T>
[[nodiscard]] inline Quaternion<T> operator *( T a, const Quaternion<T> & b )
{
    return {a * b.a, a * b.b, a * b.c, a * b.d};
}

template <typename T>
[[nodiscard]] inline Quaternion<T> operator *( const Quaternion<T> & b, T a )
{
    return {a * b.a, a * b.b, a * b.c, a * b.d};
}

template <typename T>
[[nodiscard]] inline Quaternion<T> operator /( const Quaternion<T> & b, T a )
{
    return b * ( 1 / a );
}

/// dot product
template <typename T>
[[nodiscard]] inline T dot( const Quaternion<T> & a, const Quaternion<T> & b )
{
    return a.a * b.a + a.b * b.b + a.c * b.c + a.d * b.d;
}

/// Hamilton product
template <typename T>
[[nodiscard]] inline Quaternion<T> operator *( const Quaternion<T> & q1, const Quaternion<T> & q2 )
{
    return Quaternion<T>
    {
        q1.a * q2.a - q1.b * q2.b - q1.c * q2.c - q1.d * q2.d,
        q1.a * q2.b + q1.b * q2.a + q1.c * q2.d - q1.d * q2.c,
        q1.a * q2.c - q1.b * q2.d + q1.c * q2.a + q1.d * q2.b,
        q1.a * q2.d + q1.b * q2.c - q1.c * q2.b + q1.d * q2.a
    };
}

template <typename T>
inline constexpr Vector3<T> Quaternion<T>::operator()( const Vector3<T> & p ) const noexcept
{
    // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
    return ( *this  * Quaternion( T(0), p ) * this->conjugate() ).im();
}

template<typename T>
[[nodiscard]] const Quaternion<T>* getCanonicalQuaternions() noexcept
{
    static Quaternion<T> canonQuats[24] =
    {
        Quaternion<T>(),

        Quaternion<T>( Vector3<T>::plusX(), T( PI2 ) ),
        Quaternion<T>( Vector3<T>::plusX(), T( PI ) ),
        Quaternion<T>( Vector3<T>::plusX(), T( 3 * PI2 ) ),
        Quaternion<T>( Vector3<T>::plusY(), T( PI2 ) ),
        Quaternion<T>( Vector3<T>::plusY(), T( PI ) ),
        Quaternion<T>( Vector3<T>::plusY(), T( 3 * PI2 ) ),
        Quaternion<T>( Vector3<T>::plusZ(), T( PI2 ) ),
        Quaternion<T>( Vector3<T>::plusZ(), T( PI ) ),
        Quaternion<T>( Vector3<T>::plusZ(), T( 3 * PI2 ) ),

        Quaternion<T>( Vector3<T>( T( 1 ),T( 1 ),T( 0 ) ), T( PI ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( -1 ),T( 0 ) ), T( PI ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( 0 ),T( 1 ) ), T( PI ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( 0 ),T( -1 ) ), T( PI ) ),
        Quaternion<T>( Vector3<T>( T( 0 ),T( 1 ),T( 1 ) ), T( PI ) ),
        Quaternion<T>( Vector3<T>( T( 0 ),T( 1 ),T( -1 ) ), T( PI ) ),

        Quaternion<T>( Vector3<T>( T( 1 ),T( 1 ),T( 1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( 1 ),T( -1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( -1 ),T( 1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( 1 ),T( -1 ),T( -1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( -1 ),T( 1 ),T( 1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( -1 ),T( 1 ),T( -1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( -1 ),T( -1 ),T( 1 ) ), T( 2 * PI / 3 ) ),
        Quaternion<T>( Vector3<T>( T( -1 ),T( -1 ),T( -1 ) ), T( 2 * PI / 3 ) )
    };
    return canonQuats;
}

/// returns closest to base canonical quaternion
template<typename T>
[[nodiscard]] Quaternion<T> getClosestCanonicalQuaternion( const Quaternion<T>& base ) noexcept
{
    Quaternion<T> baseInverse = base.normalized().inverse();
    int closestIndex{0};
    T maxCos = T(-2);
    const Quaternion<T>* canonQuats = getCanonicalQuaternions<T>();
    for ( int i = 0; i < 24; ++i )
    {
        const Quaternion<T>& canonQuat = canonQuats[i];
        Quaternion<T> relativeQuat = canonQuat * baseInverse;
        relativeQuat.normalize();
        T cos = std::abs( relativeQuat.a );
        if ( cos > maxCos )
        {
            maxCos = cos;
            closestIndex = i;
        }
    }
    return canonQuats[closestIndex];
}

template <typename T>
[[nodiscard]] Matrix3<T> getClosestCanonicalMatrix( const Matrix3<T>& matrix ) noexcept
{
    Quaternion<T> closestQuat = getClosestCanonicalQuaternion( Quaternion<T>( matrix ) );
    return Matrix3<T>( closestQuat );
}

/// given t in [0,1] and two rotation matrices, interpolates them spherically and produces another rotation matrix
template <typename T>
[[nodiscard]] inline Matrix3<T> slerp( const Matrix3<T> & m0, const Matrix3<T> & m1, T t )
{
    return Quaternion<T>::slerp( m0, m1, t );
}

/// given t in [0,1] and rigid transformations, interpolates them spherically and produces another rigid transformation;
/// p is the only point that will have straight line movement during interpolation
template <typename T>
[[nodiscard]] inline AffineXf3<T> slerp( const AffineXf3<T> & xf0, const AffineXf3<T> & xf1, T t, const Vector3<T> & p = {} )
{
    return Quaternion<T>::slerp( xf0, xf1, t, p );
}

/// given any matrix, returns a close rotation matrix
template <typename T>
[[nodiscard]] inline Matrix3<T> orthonormalized( const Matrix3<T> & m )
{
    return Matrix3<T>{ Quaternion<T>{ m }.normalized() };
}

/// given any affine transformation, returns a close rigid transformation;
/// center point will be transformed to same point by both input and output transformations
template <typename T>
[[nodiscard]] inline AffineXf3<T> orthonormalized( const AffineXf3<T> & xf, const Vector3<T> & center = {} )
{
    AffineXf3<T> res;
    res.A = orthonormalized( xf.A );
    res.b = xf( center ) - res.A * center;
    return res;
}

/// \}

} // namespace MR
