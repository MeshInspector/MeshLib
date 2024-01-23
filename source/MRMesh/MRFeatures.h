#pragma once

#include "MRMesh/MRCone3.h"
#include "MRMesh/MRCylinder3.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRLineSegm3.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRSphere.h"
#include "MRMesh/MRVector3.h"

#include <cassert>
#include <optional>
#include <variant>

namespace MR::Features
{

namespace Primitives
{
    // Doubles as a point when the radius is zero.
    using Sphere = Sphere3<float>;

    struct Plane
    {
        Vector3f center;

        // This must be normalized. The sign doesn't matter.
        Vector3f normal;
    };

    //! Can have infinite length in one or two directions.
    //! The top and/or bottom can be flat or pointy.
    //! Doubles as a cylinder, line (finite or infinite), and a circle.
    struct ConeSegment
    {
        // Sanity requirements:
        // * `dir` must be normalized.
        // * Both `positiveLength` and `negativeLength` should be non-negative. They can be infinite (both or individually).
        // * If they are equal (both zero) or at least one of them is infinite, `positiveSideRadius` must be equal to `negativeSideRadius`.
        // * Both `positiveSideRadius` and `negativeSideRadius` must be non-negative.

        // This isn't necessarily the true center point. Use `centerPoint()` instead.
        Vector3f center;
        Vector3f dir; //< Must be normalized.

        //! Cap radius in the `dir` direction.
        float positiveSideRadius = 0;
        //! Cap radius in the direction opposite to `dir`.
        float negativeSideRadius = 0;

        //! Distance from the `center` to the cap in the `dir` direction.
        float positiveLength = 0;
        //! Distance from the `center` to the cap in the direction opposite to `dir`.
        float negativeLength = 0;

        // If true, the cone has no caps and no volume, and all distances (to the conical surface, that is) are positive.
        bool hollow = false;

        [[nodiscard]] bool isZeroRadius() const { return positiveSideRadius == 0 && negativeSideRadius == 0; }
        [[nodiscard]] bool isCircle() const { return positiveLength == -negativeLength && std::isfinite( positiveLength ); }

        // Returns the length. Can be infinite.
        [[nodiscard]] float length() const { return positiveLength + negativeLength; }

        // Returns the center point (unlike `center`, which can actually be off-center).
        // This doesn't work for half-infinite objects.
        [[nodiscard]] MRMESH_API Sphere centerPoint() const;

        // Extends the object to infinity in one direction. The radius in the extended direction becomes equal to the radius in the opposite direction.
        [[nodiscard]] MRMESH_API ConeSegment extendToInfinity( bool negative ) const;
        // Extends the object to infinity in both directions. This is equivalent to `.extendToInfinity(false).extendToInfinity(true)`,
        // except that calling it with `positiveSideRadius != negativeSideRadius` is illegal and triggers an assertion.
        [[nodiscard]] MRMESH_API ConeSegment extendToInfinity() const;

        // Untruncates a truncated cone. If it's not a cone at all, returns the object unchanged and triggers an assertion.
        [[nodiscard]] MRMESH_API ConeSegment untruncateCone() const;

        // Returns a finite axis. For circles, you might want to immediately `extendToInfinity()` it.
        [[nodiscard]] MRMESH_API ConeSegment axis() const;

        // Returns a center of one of the two base circles.
        [[nodiscard]] MRMESH_API Sphere basePoint( bool negative ) const;
        // Returns one of the two base planes.
        [[nodiscard]] MRMESH_API Plane basePlane( bool negative ) const;
        // Returns one of the two base circles.
        [[nodiscard]] MRMESH_API ConeSegment baseCircle( bool negative ) const;
    };

    using Variant = std::variant<Sphere, ConeSegment, Plane>;
}

// Those map various MR types to our primitives. Some of those are identity functions.

[[nodiscard]] inline Primitives::Sphere toPrimitive( const Vector3f& point ) { return { point, 0 }; }
[[nodiscard]] inline Primitives::Sphere toPrimitive( const Sphere3f& sphere ) { return sphere; }

[[nodiscard]] inline Primitives::ConeSegment toPrimitive( const Line3f& line ) { return { .center = line.p, .dir = line.d.normalized(), .positiveLength = INFINITY, .negativeLength = INFINITY }; }
[[nodiscard]] inline Primitives::ConeSegment toPrimitive( const LineSegm3f& segm ) { return { .center = segm.a, .dir = segm.dir().normalized(), .positiveLength = segm.length() }; }

[[nodiscard]] inline Primitives::ConeSegment toPrimitive( const Cylinder3f& cyl )
{
    float halfLen = cyl.length / 2;
    return{
        .center = cyl.center(),
        .dir = cyl.direction().normalized(),
        .positiveSideRadius = cyl.radius, .negativeSideRadius = cyl.radius,
        .positiveLength = halfLen, .negativeLength = halfLen,
    };
}
[[nodiscard]] inline Primitives::ConeSegment toPrimitive( const Cone3f& cone )
{
    return{
        .center = cone.center(),
        .dir = cone.direction().normalized(),
        .positiveSideRadius = std::tan( cone.angle ) * cone.height, .negativeSideRadius = 0,
        .positiveLength = cone.height, .negativeLength = 0,
    };
}

//! `normal` doesn't need to be normalized.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCircle( const Vector3f& point, const Vector3f& normal, float rad );
//! `a` and `b` are centers of the sides.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCylinder( const Vector3f& a, const Vector3f& b, float rad );
//! `a` is the center of the base, `b` is the pointy end.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCone( const Vector3f& a, const Vector3f& b, float rad );

// Returns null if the object type is unknown.
[[nodiscard]] MRMESH_API std::optional<Primitives::Variant> primitiveFromObject( const Object& object );
// Can return null on some primitive configurations.
// `infiniteExtend` is how large we make "infinite" objects. Half-infinite objects divide this by 2.
[[nodiscard]] MRMESH_API std::shared_ptr<VisualObject> primitiveToObject( const Primitives::Variant& primitive, float infiniteExtent );

//! Stores the distance between two objects, and the closest points on them.
struct DistanceResult
{
    enum class Status
    {
        ok = 0,
        //! Algorithms set this if this when something isn't yet implemented.
        not_implemented,
        //! Algorithms set this when the calculation doesn't make sense for those object types.
        not_applicable,
        //! This is set automatically if either `distance` or at least one of the points isn't finite. But you can set this from an algorithm too.
        not_finite,
    };
    Status status = Status::ok;

    // This is a separate field because it can be negative.
    float distance = 0;

    Vector3f closestPointA;
    Vector3f closestPointB;

    [[nodiscard]] operator bool() const { return status == Status::ok; }
};

//! Traits that determine how the primitives are related.
namespace Traits
{

template <typename T>
struct Unary {};
template <>
struct Unary<Primitives::Sphere>
{
    MRMESH_API std::string name( const Primitives::Sphere& prim ) const;
};
template <>
struct Unary<Primitives::ConeSegment>
{
    MRMESH_API std::string name( const Primitives::ConeSegment& prim ) const;
};
template <>
struct Unary<Primitives::Plane>
{
    MRMESH_API std::string name( const Primitives::Plane& prim ) const;
};

template <typename A, typename B>
struct Binary {};

//! Whether we have traits to get distance from A to B.
//! This is for internal use because it's asymmetrical, we a have a symmetrical version below.
template <typename A, typename B>
concept DistanceSupportedOneWay = requires( const Binary<A, B>& t, const A& a, const B& b )
{
    { t.distance( a, b ) } -> std::same_as<DistanceResult>;
};

// ?? <-> Sphere
template <>
struct Binary<Primitives::Sphere, Primitives::Sphere>
{
    MRMESH_API DistanceResult distance( const Primitives::Sphere& a, const Primitives::Sphere& b ) const;
};
template <>
struct Binary<Primitives::ConeSegment, Primitives::Sphere>
{
    MRMESH_API DistanceResult distance( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const;
};
template <>
struct Binary<Primitives::Plane, Primitives::Sphere>
{
    MRMESH_API DistanceResult distance( const Primitives::Plane& a, const Primitives::Sphere& b ) const;
};

// ?? <-> Cone
template <>
struct Binary<Primitives::ConeSegment, Primitives::ConeSegment>
{
    MRMESH_API DistanceResult distance( const Primitives::ConeSegment& a, const Primitives::ConeSegment& b ) const;
};
template <>
struct Binary<Primitives::Plane, Primitives::ConeSegment>
{
    MRMESH_API DistanceResult distance( const Primitives::Plane& a, const Primitives::ConeSegment& b ) const;
};

// ?? <-> Plane
template <>
struct Binary<Primitives::Plane, Primitives::Plane>
{
    MRMESH_API DistanceResult distance( const Primitives::Plane& a, const Primitives::Plane& b ) const;
};

}

template <typename T>
[[nodiscard]] std::string name( const T& primitive )
{
    return Traits::Unary<T>{}.name( primitive );
}

// Whether the distance can be computed between primitives.
template <typename A, typename B>
concept DistanceSupported = Traits::DistanceSupportedOneWay<A, B> || Traits::DistanceSupportedOneWay<B, A>;

// Computes the distance between two primitives.
template <typename A, typename B>
requires DistanceSupported<A, B>
[[nodiscard]] DistanceResult distance( const A& a, const B& b )
{
    if constexpr ( Traits::DistanceSupportedOneWay<A, B> )
    {
        DistanceResult ret = Traits::Binary<A, B>{}.distance( a, b );

        if ( ret && ( !std::isfinite( ret.distance ) || !ret.closestPointA.isFinite() || !ret.closestPointB.isFinite() ) )
        {
            ret.status = DistanceResult::Status::not_finite;
            return ret;
        }

        // Check that we got the correct distance here.
        // Note that the distance is signed, so we apply `abs` to it to compare it properly.
        assert( std::abs( ( ret.closestPointB - ret.closestPointA ).length() - std::abs( ret.distance ) ) < 0.0001f );

        return ret;
    }
    else
    {
        static_assert( Traits::DistanceSupportedOneWay<B, A>, "This should never fail." );
        DistanceResult ret = (distance)( b, a );
        std::swap( ret.closestPointA, ret.closestPointB );
        return ret;
    }
}

}