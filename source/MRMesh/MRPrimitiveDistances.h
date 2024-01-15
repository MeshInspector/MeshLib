#pragma once

#include "MRMesh/MRLine3.h"
#include "MRMesh/MRLineSegm3.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRSphere.h"
#include "MRMesh/MRVector3.h"

namespace MR::PrimitiveDistances
{

namespace Primitives
{
    // Doubles as a point when the radius is zero.
    using Sphere = Sphere3<float>;

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
    };

    // This doesn't have to be normalized, we will normalize it internally.
    // The sign of the normal doesn't matter.
    using Plane = Plane3<float>;
}

// Those map various MR types to our primitives. Some of those are identity functions.

[[nodiscard]] inline Primitives::Sphere toPrimitive( Vector3f point ) { return { point, 0 }; }
[[nodiscard]] inline Primitives::Sphere toPrimitive( Sphere3<float> sphere ) { return sphere; }

[[nodiscard]] inline Primitives::ConeSegment toPrimitive( Line3f line ) { return { .center = line.p, .dir = line.d.normalized(), .positiveLength = INFINITY, .negativeLength = INFINITY }; }
[[nodiscard]] inline Primitives::ConeSegment toPrimitive( LineSegm3f segm ) { return { .center = segm.a, .dir = segm.dir().normalized(), .positiveLength = segm.length() }; }

[[nodiscard]] inline Primitives::Plane toPrimitive( Plane3<float> plane ) { return plane; }

//! `normal` doesn't need to be normalized.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCircle( Vector3f point, Vector3f normal, float rad );
//! `a` and `b` are centers of the sides.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCylinder( Vector3f a, Vector3f b, float rad );
//! `a` is the center of the base, `b` is the pointy end.
[[nodiscard]] MRMESH_API Primitives::ConeSegment primitiveCone( Vector3f a, Vector3f b, float rad );
//! Constructs a plane from a point and a normal. Or you could construct the plane type directly from a normal and an offset.
[[nodiscard]] MRMESH_API Primitives::Plane primitivePlane( Vector3f point, Vector3f normal );

//! Stores the distance between two objects, and the closest points on them.
struct DistanceResult
{
    enum class Status
    {
        ok = 0,
        // Algorithms set this if this specific configuration of primitives is unsupported.
        unsupported,
        // This is set automatically if either `distance` or at least one of the points isn't finite. But you can set this from an algorithm too.
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

template <typename A, typename B>
struct Distance {};

//! Whether we have traits to get distance from A to B.
//! This is for internal use because it's asymmetrical, we a have a symmetrical version below.
template <typename A, typename B>
concept DistanceSupportedOneWay = requires( const Distance<A, B>& t, const A& a, const B& b )
{
    { t( a, b ) } -> std::same_as<DistanceResult>;
};

// ?? <-> Sphere
template <>
struct Distance<Primitives::Sphere, Primitives::Sphere>
{
    MRMESH_API DistanceResult operator()( const Primitives::Sphere& a, const Primitives::Sphere& b ) const;
};
template <>
struct Distance<Primitives::ConeSegment, Primitives::Sphere>
{
    MRMESH_API DistanceResult operator()( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const;
};
template <>
struct Distance<Primitives::Plane, Primitives::Sphere>
{
    MRMESH_API DistanceResult operator()( const Primitives::Plane& a, const Primitives::Sphere& b ) const;
};

// ?? <-> Cone
template <>
struct Distance<Primitives::ConeSegment, Primitives::ConeSegment>
{
    MRMESH_API DistanceResult operator()( const Primitives::ConeSegment& a, const Primitives::ConeSegment& b ) const;
};
template <>
struct Distance<Primitives::Plane, Primitives::ConeSegment>
{
    MRMESH_API DistanceResult operator()( const Primitives::Plane& a, const Primitives::ConeSegment& b ) const;
};

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
        DistanceResult ret = Traits::Distance<A, B>{}( a, b );

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
        DistanceResult ret = Traits::Distance<B, A>{}( b, a );
        std::swap( ret.closestPointA, ret.closestPointB );
        return ret;
    }
}

}