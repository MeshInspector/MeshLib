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

namespace MR
{
class FeatureObject;
}

namespace MR::Features
{

namespace Primitives
{
    struct Plane;
    struct ConeSegment;

    // ---

    // Doubles as a point when the radius is zero.
    using Sphere = Sphere3<float>;

    struct Plane
    {
        Vector3f center;

        // This must be normalized. The sign doesn't matter.
        Vector3f normal = Vector3f( 1, 0, 0 );

        // Returns an infinite line, with the center in a sane location.
        [[nodiscard]] MRMESH_API ConeSegment intersectWithPlane( const Plane& other ) const;

        // Intersects the plane with a line, returns a point (zero radius sphere).
        // Only `center` and `dir` are used from `line` (so if `line` is a cone/cylinder, its axis is used,
        // and the line is extended to infinity).
        [[nodiscard]] MRMESH_API Sphere intersectWithLine( const ConeSegment& line ) const;

        friend bool operator==( const Plane&, const Plane& ) = default;
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

        friend bool operator==( const ConeSegment&, const ConeSegment& ) = default;

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
// `infiniteExtent` is how large we make "infinite" objects. Half-infinite objects divide this by 2.
[[nodiscard]] MRMESH_API std::shared_ptr<FeatureObject> primitiveToObject( const Primitives::Variant& primitive, float infiniteExtent );

//! Stores the results of measuring two objects relative to one another.
struct MeasureResult
{
    enum class Status
    {
        ok = 0,
        //! Algorithms set this if this when something isn't yet implemented.
        notImplemented,
        //! Algorithms set this when the calculation doesn't make sense for those object types.
        //! This result can be based on object parameters, but not on their relative location.
        badFeaturePair,
        //! Can't be computed because of how the objects are located relative to each other.
        badRelativeLocation,
        //! The result was not finite. This is set automatically if you return non-finite values, but you can also set this manually.
        notFinite,
    };

    struct BasicPart
    {
        Status status = Status::notImplemented;
        [[nodiscard]] operator bool() const { return status == Status::ok; }
    };

    struct Distance : BasicPart
    {
        // This is a separate field because it can be negative.
        float distance = 0;

        Vector3f closestPointA;
        Vector3f closestPointB;

        [[nodiscard]] Vector3f closestPointFor( bool b ) const { return b ? closestPointB : closestPointA; }
    };
    Distance distance;

    struct Angle : BasicPart
    {
        Vector3f pointA;
        Vector3f pointB;
        [[nodiscard]] Vector3f pointFor( bool b ) const { return b ? pointB : pointA; }

        Vector3f dirA; // Normalized.
        Vector3f dirB; // ^
        [[nodiscard]] Vector3f dirFor( bool b ) const { return b ? dirB : dirA; }

        /// Whether `dir{A,B}` is a surface normal or a line direction.
        bool isSurfaceNormalA = false;
        bool isSurfaceNormalB = false;

        [[nodiscard]] bool isSurfaceNormalFor( bool b ) const { return b ? isSurfaceNormalB : isSurfaceNormalA; }

        [[nodiscard]] MRMESH_API float computeAngleInRadians() const;
    };
    Angle angle;

    // The primitives obtained from intersecting those two.
    std::vector<Primitives::Variant> intersections;

    // Modifies the object to swap A and B;
    MRMESH_API void swapObjects();
};
// `MeasureResult::Status` enum to string.
[[nodiscard]] MRMESH_API std::string_view toString( MeasureResult::Status status );

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
concept MeasureSupportedOneWay = requires( const Binary<A, B>& t, const A& a, const B& b )
{
    { t.measure( a, b ) } -> std::same_as<MeasureResult>;
};

// ?? <-> Sphere
template <>
struct Binary<Primitives::Sphere, Primitives::Sphere>
{
    MRMESH_API MeasureResult measure( const Primitives::Sphere& a, const Primitives::Sphere& b ) const;
};
template <>
struct Binary<Primitives::ConeSegment, Primitives::Sphere>
{
    MRMESH_API MeasureResult measure( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const;
};
template <>
struct Binary<Primitives::Plane, Primitives::Sphere>
{
    MRMESH_API MeasureResult measure( const Primitives::Plane& a, const Primitives::Sphere& b ) const;
};

// ?? <-> Cone
template <>
struct Binary<Primitives::ConeSegment, Primitives::ConeSegment>
{
    MRMESH_API MeasureResult measure( const Primitives::ConeSegment& a, const Primitives::ConeSegment& b ) const;
};
template <>
struct Binary<Primitives::Plane, Primitives::ConeSegment>
{
    MRMESH_API MeasureResult measure( const Primitives::Plane& a, const Primitives::ConeSegment& b ) const;
};

// ?? <-> Plane
template <>
struct Binary<Primitives::Plane, Primitives::Plane>
{
    MRMESH_API MeasureResult measure( const Primitives::Plane& a, const Primitives::Plane& b ) const;
};

}

// Get name of a `Primitives::...` class (can depend on its parameters).
template <typename T>
[[nodiscard]] std::string name( const T& primitive )
{
    return Traits::Unary<T>{}.name( primitive );
}
// Same but for a variant.
[[nodiscard]] MRMESH_API std::string name( const Primitives::Variant& var );

// Whether you can measure two primitives relative to one another.
template <typename A, typename B>
concept MeasureSupported = Traits::MeasureSupportedOneWay<A, B> || Traits::MeasureSupportedOneWay<B, A>;

// Measures stuff between two primitives. (Two types from `Primitives::...`.)
template <typename A, typename B>
requires MeasureSupported<A, B>
[[nodiscard]] MeasureResult measure( const A& a, const B& b )
{
    if constexpr ( Traits::MeasureSupportedOneWay<A, B> )
    {
        MeasureResult ret = Traits::Binary<A, B>{}.measure( a, b );

        // Catch non-finite distance.
        if ( ret.distance && ( !std::isfinite( ret.distance.distance ) || !ret.distance.closestPointA.isFinite() || !ret.distance.closestPointB.isFinite() ) )
            ret.distance.status = MeasureResult::Status::badRelativeLocation;

        // Check that we got the correct distance here.
        // Note that the distance is signed, so we apply `abs` to it to compare it properly.
        assert( ret.distance <= ( std::abs( ( ret.distance.closestPointB - ret.distance.closestPointA ).length() - std::abs( ret.distance.distance ) ) < 0.0001f ) );

        // Catch non-finite angle.
        if ( ret.angle && ( !ret.angle.pointA.isFinite() || !ret.angle.pointB.isFinite() || !ret.angle.dirA.isFinite() || !ret.angle.dirB.isFinite() ) )
            ret.angle.status = MeasureResult::Status::badRelativeLocation;

        // Check that the angle normals are normalized.
        assert( ret.angle <= ( std::abs( 1 - ret.angle.dirA.length() ) < 0.0001f ) );
        assert( ret.angle <= ( std::abs( 1 - ret.angle.dirB.length() ) < 0.0001f ) );

        return ret;
    }
    else
    {
        static_assert( Traits::MeasureSupportedOneWay<B, A>, "This should never fail." );
        MeasureResult ret = ( measure )( b, a );
        ret.swapObjects();
        return ret;
    }
}
// Same, but with a variant as the first argument.
template <typename B>
[[nodiscard]] MeasureResult measure( const Primitives::Variant& a, const B& b )
{
    return std::visit( [&]( const auto& elem ){ return (measure)( elem, b ); }, a );
}
// Same, but with a variant as the second argument.
template <typename A>
[[nodiscard]] MeasureResult measure( const A& a, const Primitives::Variant& b )
{
    return std::visit( [&]( const auto& elem ){ return (measure)( a, elem ); }, b );
}
// Same, but with variants as both argument.
[[nodiscard]] MRMESH_API MeasureResult measure( const Primitives::Variant& a, const Primitives::Variant& b );

}
