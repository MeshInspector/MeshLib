#include "MRFeatures.h"

#include "MRMesh/MRGTest.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"

// The tests in this file should be checked in debug builds too!

namespace MR::Features
{

Primitives::ConeSegment Primitives::Plane::intersectWithPlane( const Plane& other ) const
{
    Vector3f point = intersectWithLine( { .referencePoint = other.center, .dir = cross( other.normal, cross( other.normal, normal ) ).normalized() } ).center;
    return toPrimitive( Line( point, cross( normal, other.normal ) ) );
}

Primitives::Sphere Primitives::Plane::intersectWithLine( const ConeSegment& line ) const
{
    return toPrimitive( line.referencePoint - dot( line.referencePoint - center, normal ) / dot( line.dir, normal ) * line.dir );
}

Primitives::Sphere Primitives::ConeSegment::centerPoint() const
{
    bool posFinite = std::isfinite( positiveLength );
    bool negFinite = std::isfinite( negativeLength );

    if ( posFinite != negFinite )
        return basePoint( negFinite );

    return toPrimitive( posFinite && negFinite ? referencePoint + dir * ( ( positiveLength - negativeLength ) / 2 ) : referencePoint );
}

Primitives::ConeSegment Primitives::ConeSegment::extendToInfinity( bool negative ) const
{
    ConeSegment ret = *this;
    if ( negative )
    {
        ret.negativeSideRadius = ret.positiveSideRadius;
        ret.negativeLength = INFINITY;
    }
    else
    {
        ret.positiveSideRadius = ret.negativeSideRadius;
        ret.positiveLength = INFINITY;
    }
    return ret;
}

Primitives::ConeSegment Primitives::ConeSegment::extendToInfinity() const
{
    assert( positiveSideRadius == negativeSideRadius );
    return extendToInfinity( false ).extendToInfinity( true );
}

Primitives::ConeSegment Primitives::ConeSegment::untruncateCone() const
{
    bool isCone = !isCircle() && positiveSideRadius != negativeSideRadius;
    assert( isCone );
    if ( !isCone )
        return *this; // Not a cone.

    if ( positiveSideRadius == 0 || negativeSideRadius == 0 )
        return *this; // Already truncated.

    ConeSegment ret = *this;

    float shift = std::min( positiveSideRadius, negativeSideRadius ) * length() / std::abs( positiveSideRadius - negativeSideRadius );
    ( positiveSideRadius < negativeSideRadius ? ret.positiveLength : ret.negativeLength ) += shift;
    return ret;
}

Primitives::ConeSegment Primitives::ConeSegment::axis() const
{
    ConeSegment ret = *this;
    ret.positiveSideRadius = ret.negativeSideRadius = 0;
    return ret;
}

Primitives::Sphere Primitives::ConeSegment::basePoint( bool negative ) const
{
    return toPrimitive( referencePoint + dir * ( negative ? -negativeLength : positiveLength ) );
}

Primitives::Plane Primitives::ConeSegment::basePlane( bool negative ) const
{
    assert( std::isfinite( negative ? negativeLength : positiveLength ) );
    return { .center = basePoint( negative ).center, .normal = negative ? -dir : dir };
}

Primitives::ConeSegment Primitives::ConeSegment::baseCircle( bool negative ) const
{
    assert( std::isfinite( negative ? negativeLength : positiveLength ) );

    ConeSegment ret = *this;
    ret.referencePoint = basePoint( negative ).center;
    ret.positiveLength = ret.negativeLength = 0;
    if ( negative )
        ret.positiveSideRadius = ret.negativeSideRadius;
    else
        ret.negativeSideRadius = ret.positiveSideRadius;
    if ( negative )
        ret.dir = -ret.dir;
    return ret;
}

Primitives::ConeSegment primitiveCircle( const Vector3f& point, const Vector3f& normal, float rad )
{
    return {
        .referencePoint = point,
        .dir = normal.normalized(),
        .positiveSideRadius = rad,
        .negativeSideRadius = rad,
    };
}

Primitives::ConeSegment primitiveCylinder( const Vector3f& a, const Vector3f& b, float rad )
{
    auto ret = primitiveCone( a, b, rad );
    ret.positiveSideRadius = ret.negativeSideRadius;
    return ret;
}

Primitives::ConeSegment primitiveCone( const Vector3f& a, const Vector3f& b, float rad )
{
    Vector3f delta = b - a;
    float deltaLen = delta.length();
    return {
        .referencePoint = a,
        .dir = delta / ( deltaLen > 0 ? deltaLen : 1 ),
        .negativeSideRadius = rad,
        .positiveLength = deltaLen,
    };
}

std::optional<Primitives::Variant> primitiveFromObject( const Object& object )
{
    // FIXME: We use `getUniformScale` in a few place before, but the scale isn't always uniform!
    // E.g. when we're a child of a different feature.

    // Extracts a uniform scale factor from a matrix.
    // If the scaling isn't actually uniform, returns some unspecified average scaling, which is hopefully better than just taking an arbitrary axis.
    static constexpr auto getUniformScale = [&]( const Matrix3f& m ) -> float
    {
        Matrix3f r, s;
        decomposeMatrix3( m, r, s );
        return ( s.x.x + s.y.y + s.z.z ) / 3;
    };

    AffineXf3f parentXf;
    if ( object.parent() )
        parentXf = object.parent()->worldXf(); // Otherwise an identity xf is used, which is fine.

    if ( auto point = dynamic_cast<const PointObject*>( &object ) )
    {
        return toPrimitive( parentXf( point->getPoint() ) );
    }
    else if ( auto line = dynamic_cast<const LineObject*>( &object ) )
    {
        return toPrimitive( LineSegm3f( parentXf( line->getPointA() ), parentXf( line->getPointB() ) ) );
    }
    else if ( auto plane = dynamic_cast<const PlaneObject*>( &object ) )
    {
        return Primitives::Plane{ .center = parentXf( plane->getCenter() ), .normal = ( parentXf.A * plane->getNormal() ).normalized() };
    }
    else if ( auto sphere = dynamic_cast<const SphereObject*>( &object ) )
    {
        return toPrimitive( Sphere( parentXf( sphere->getCenter() ), sphere->getRadius() * getUniformScale( parentXf.A ) ) );
    }
    else if ( auto circle = dynamic_cast<const CircleObject*>( &object ) )
    {
        float radius = circle->getRadius() * getUniformScale( parentXf.A );
        return Primitives::ConeSegment{
            .referencePoint = parentXf( circle->getCenter() ),
            .dir = parentXf.A * circle->getNormal(),
            .positiveSideRadius = radius,
            .negativeSideRadius = radius,
            .hollow = true,
        };
    }
    else if ( auto cyl = dynamic_cast<const CylinderObject*>( &object ) )
    {
        float scale = getUniformScale( parentXf.A );
        float radius = cyl->getRadius() * scale;
        float halfLen = cyl->getLength() / 2 * scale;
        return Primitives::ConeSegment{
            .referencePoint = parentXf( cyl->getCenter() ),
            .dir = parentXf.A * cyl->getDirection(),
            .positiveSideRadius = radius,
            .negativeSideRadius = radius,
            .positiveLength = halfLen,
            .negativeLength = halfLen,
            .hollow = true, // I guess?
        };
    }
    else if ( auto cone = dynamic_cast<const ConeObject*>( &object ) )
    {
        // I want the "positive" direction to point towards the tip (so "axis -> positive/negative end").
        // It's moot where the center should be, but currently having it at the tip is ok.
        Primitives::ConeSegment ret{
            .referencePoint = parentXf( cone->getCenter() ),
            .dir = parentXf.A * -cone->getDirection(),
            .positiveSideRadius = 0,
            .negativeSideRadius = cone->getBaseRadius() * getUniformScale( parentXf.A ),
            .positiveLength = 0,
            .negativeLength = cone->getHeight(),
            .hollow = true, // I guess?
        };
        return ret;
    }

    return {};
}

std::shared_ptr<FeatureObject> primitiveToObject( const Primitives::Variant& primitive, float infiniteExtent )
{
    return std::visit( overloaded{
        []( const Primitives::Sphere& sphere ) -> std::shared_ptr<FeatureObject>
        {
            if ( sphere.radius == 0 )
            {
                auto newPoint = std::make_shared<PointObject>();
                newPoint->setPoint( sphere.center );
                return newPoint;
            }
            else
            {
                auto newSphere = std::make_shared<SphereObject>();
                newSphere->setCenter( sphere.center );
                newSphere->setRadius( sphere.radius );
                return newSphere;
            }
        },
        [infiniteExtent]( const Primitives::Plane& plane ) -> std::shared_ptr<FeatureObject>
        {
            auto newPlane = std::make_shared<PlaneObject>();
            newPlane->setCenter( plane.center );
            newPlane->setNormal( plane.normal );
            newPlane->setSize( infiniteExtent );
            return newPlane;
        },
        [infiniteExtent]( const Primitives::ConeSegment& cone ) -> std::shared_ptr<FeatureObject>
        {
            if ( cone.isCircle() )
            {
                if ( cone.isZeroRadius() )
                    return primitiveToObject( cone.basePoint( false ), infiniteExtent ); // This isn't really valid, but let's support it.

                auto newCircle = std::make_shared<CircleObject>();
                newCircle->setCenter( cone.basePoint( false ).center );
                newCircle->setNormal( cone.dir );
                newCircle->setRadius( cone.positiveSideRadius );
                return newCircle;
            }

            bool posFinite = std::isfinite( cone.positiveLength );
            bool negFinite = std::isfinite( cone.negativeLength );

            if ( cone.isZeroRadius() )
            {
                auto newLine = std::make_shared<LineObject>();
                newLine->setDirection( cone.dir );

                if ( posFinite == negFinite )
                {
                    // Finite or fully infinite.
                    newLine->setCenter( cone.centerPoint().center );
                    newLine->setLength( posFinite ? cone.length() : infiniteExtent );
                }
                else
                {
                    // Half-infinite.
                    newLine->setCenter( posFinite
                        ? cone.basePoint( false ).center - cone.dir * ( infiniteExtent / 2 )
                        : cone.basePoint( true ).center + cone.dir * ( infiniteExtent / 2 )
                    );
                    newLine->setLength( infiniteExtent );
                }

                return newLine;
            }

            if ( cone.positiveSideRadius == cone.negativeSideRadius )
            {
                auto newCylinder = std::make_shared<CylinderObject>();
                newCylinder->setDirection( cone.dir );
                newCylinder->setRadius( cone.positiveSideRadius );

                if ( posFinite == negFinite )
                {
                    newCylinder->setCenter( cone.centerPoint().center );
                    newCylinder->setLength( posFinite ? cone.length() : infiniteExtent );
                }
                else
                {
                    newCylinder->setCenter( posFinite
                        ? cone.basePoint( false ).center - cone.dir * ( infiniteExtent / 2 )
                        : cone.basePoint( true ).center + cone.dir * ( infiniteExtent / 2 )
                    );
                    newCylinder->setLength( infiniteExtent );
                }

                return newCylinder;
            }

            if ( cone.positiveSideRadius == 0 || cone.negativeSideRadius == 0 )
            {
                // A non-truncated cone.

                bool flip = cone.positiveSideRadius == 0;

                // Sanity check.
                if ( flip ? posFinite : negFinite )
                {
                    auto newCone = std::make_shared<ConeObject>();
                    newCone->setCenter( cone.basePoint( !flip ).center );
                    newCone->setDirection( cone.dir * ( flip ? -1.f : 1.f ) );
                    newCone->setHeight( ( flip ? negFinite : posFinite ) ? cone.length() : infiniteExtent / 2 );
                    newCone->setBaseRadius( flip ? cone.negativeSideRadius : cone.positiveSideRadius );
                    return newCone;
                }
            }

            // TODO support truncated cones, when we have an object for them?
            return nullptr;
        },
    }, primitive );
}

float MeasureResult::Angle::computeAngleInRadians() const
{
    float ret = std::acos( std::clamp( dot( dirA, dirB ), -1.f, 1.f ) );
    if ( isSurfaceNormalA != isSurfaceNormalB )
        ret = MR::PI2_F - ret;
    return ret;
}

void MeasureResult::swapObjects()
{
    std::swap( distance.closestPointA, distance.closestPointB );
    std::swap( angle.pointA, angle.pointB );
    std::swap( angle.dirA, angle.dirB );
    std::swap( angle.isSurfaceNormalA, angle.isSurfaceNormalB );
}

std::string_view toString( MeasureResult::Status status )
{
    switch ( status )
    {
    case Features::MeasureResult::Status::ok:
        return "Ok";
        break;
    case Features::MeasureResult::Status::notImplemented:
        return "Sorry, not implemented yet for those features";
        break;
    case Features::MeasureResult::Status::badFeaturePair:
        return "Doesn't make sense for those features";
        break;
    case Features::MeasureResult::Status::badRelativeLocation:
        return "N/A because of how the features are located";
        break;
    case Features::MeasureResult::Status::notFinite:
        return "Infinite";
        break;
    }
    assert( false && "Invalid enum." );
    return "??";
}

namespace Traits
{

std::string Unary<Primitives::Sphere>::name( const Primitives::Sphere& prim ) const
{
    if ( prim.radius == 0 )
        return "Point";
    else
        return "Sphere";
}

std::string Unary<Primitives::ConeSegment>::name( const Primitives::ConeSegment& prim ) const
{
    if ( prim.isCircle() )
        return "Circle";

    if ( prim.positiveSideRadius == prim.negativeSideRadius )
    {
        int numInf = !std::isfinite( prim.positiveLength ) + !std::isfinite( prim.negativeLength );

        if ( prim.positiveSideRadius == 0 )
            return std::array{ "Line segment", "Ray", "Line" }[numInf];
        else
            return std::array{ "Cylinder", "Half-infinite cylinder", "Infinite cylinder" }[numInf];
    }

    if ( prim.positiveSideRadius == 0 || prim.negativeSideRadius == 0 )
        return "Cone";
    else
        return "Truncated cone";
}

std::string Unary<Primitives::Plane>::name( const Primitives::Plane& prim ) const
{
    (void)prim;
    return "Plane";
}

MeasureResult Binary<Primitives::Sphere, Primitives::Sphere>::measure( const Primitives::Sphere& a, const Primitives::Sphere& b ) const
{
    MeasureResult ret;

    // Distance.

    Vector3f dir = b.center - a.center;
    float dirLen = dir.length();
    ret.distance.status = MeasureResult::Status::ok;
    ret.distance.distance = dirLen - a.radius - b.radius;
    if ( dirLen > 0 )
        dir /= dirLen;
    else
        dir = Vector3f( 1, 0, 0 ); // An arbitrary default direction.
    ret.distance.closestPointA = a.center + a.radius * dir;
    ret.distance.closestPointB = b.center - b.radius * dir;

    // Angle.

    if ( a.radius == 0 || b.radius == 0 )
    {
        ret.angle.status = MeasureResult::Status::badFeaturePair;
    }
    else
    {
        float s = ( dirLen + a.radius + b.radius ) / 2; // https://en.wikipedia.org/wiki/Altitude_(triangle)
        float intersectionSideOffset = std::sqrt( s * ( s - dirLen ) * ( s - a.radius ) * ( s - b.radius ) ) * 2 / dirLen;
        if ( !std::isfinite( intersectionSideOffset ) )
        {
            ret.angle.status = MeasureResult::Status::badRelativeLocation;
        }
        else
        {
            ret.angle.status = MeasureResult::Status::ok;

            float intersectionFwdOffset = std::sqrt( a.radius * a.radius - intersectionSideOffset * intersectionSideOffset );
            Vector3f sideDir = cross( dir, dir.furthestBasisVector() ).normalized();

            ret.angle.pointA = ret.angle.pointB = a.center + dir * intersectionFwdOffset + sideDir * intersectionSideOffset;
            ret.angle.dirA = ( ret.angle.pointA - a.center ).normalized();
            ret.angle.dirB = ( ret.angle.pointB - b.center ).normalized();
            ret.angle.isSurfaceNormalA = ret.angle.isSurfaceNormalB = true;

            // The circle at the intersection.
            ret.intersections.push_back( primitiveCircle( a.center + dir * intersectionFwdOffset, dir, intersectionSideOffset ) );
        }
    }

    // Center distance.
    ret.centerDistance.status = MeasureResult::Status::ok;
    ret.centerDistance.distance = dirLen;
    ret.centerDistance.closestPointA = a.center;
    ret.centerDistance.closestPointB = b.center;

    return ret;
}

MeasureResult Binary<Primitives::ConeSegment, Primitives::Sphere>::measure( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const
{
    Vector3f centerDelta = b.center - a.referencePoint;

    float coneLength = a.negativeLength + a.positiveLength;
    bool coneLengthIsFinite = std::isfinite( coneLength );

    float coneRadiusDelta = a.negativeSideRadius - a.positiveSideRadius;

    // Signed distance of sphere center along the cone axis, relative to its center.
    float signedDistAlongAxis = dot( centerDelta, a.dir );

    // Delta from the cylinder axis (orthogonal to it) towards the sphere center.
    Vector3f axisToSphereCenterDelta = centerDelta - a.dir * signedDistAlongAxis;
    float axisToSphereCenterDist = axisToSphereCenterDelta.length();
    Vector3f axisToSphereCenterDir = axisToSphereCenterDelta;
    if ( axisToSphereCenterDist > 0 )
    {
        axisToSphereCenterDir /= axisToSphereCenterDist;

        // Make orthogonal to the cone axis to increase stability.
        axisToSphereCenterDir = cross( a.dir, cross( axisToSphereCenterDir, a.dir ) ).normalized();
    }
    else
    {
        axisToSphereCenterDir = cross( a.dir, a.dir.furthestBasisVector() ).normalized(); // An arbitrary direction.
    }

    // Direction parallel to the conical surface, from the sphere center towards the positive cone tip.
    Vector3f dirToPositiveTip;
    float dirToPositiveTipOriginalLen;
    if ( coneLengthIsFinite )
    {
        dirToPositiveTip = a.dir * coneLength - axisToSphereCenterDir * coneRadiusDelta;
        dirToPositiveTipOriginalLen = dirToPositiveTip.length();
        dirToPositiveTip /= dirToPositiveTipOriginalLen; // Not sure if we need to check for zero here.
    }
    else
    {
        dirToPositiveTip = a.dir;
        dirToPositiveTipOriginalLen = 1;
    }

    // Ratio of the length along the conical surface to the same length projected onto the cone axis.
    float lengthFac = 1.f / dot( dirToPositiveTip, a.dir );

    // Normal from the conical surface to the sphere center.
    Vector3f normalToConicalSurface = coneLengthIsFinite
        ? ( a.dir * coneRadiusDelta + axisToSphereCenterDir * coneLength ) / dirToPositiveTipOriginalLen
        : axisToSphereCenterDir;

    // A point on the cone axis where a normal to the conical surface hits the positive cap edge.
    float positiveCapPos = a.positiveLength - a.positiveSideRadius * coneRadiusDelta / coneLength;
    // A point on the cone axis where a normal to the conical surface hits the negative cap edge.
    float negativeCapPos = -a.negativeLength - a.negativeSideRadius * coneRadiusDelta / coneLength;

    // A point on the cone axis where a normal to the conical surface hits the sphere center.
    float projectedSpherePos = signedDistAlongAxis - axisToSphereCenterDist * coneRadiusDelta / coneLength;

    // Signed distance from the sphere center to the positive cap edge, measured in parallel to the conical surface (positive if beyond the edge).
    float slopedSignedDistToPositiveCap = ( projectedSpherePos - positiveCapPos ) / lengthFac;
    // Signed distance from the sphere center to the negative cap edge, measured in parallel to the conical surface (positive if beyond the edge).
    float slopedSignedDistToNegativeCap = ( negativeCapPos - projectedSpherePos ) / lengthFac;
    // Whether the positive cap edge is closer to the sphere center than the negative one, measured in parallel to the conical surface.
    bool positiveCapIsSlopedCloser = slopedSignedDistToPositiveCap > slopedSignedDistToNegativeCap;

    // Distance between the conical surface and the cone axis, measured along the normal from the conical surface to the spehre center.
    float axisToSurfaceSlopedDist = coneLengthIsFinite
        ? ( ( projectedSpherePos - negativeCapPos ) / ( positiveCapPos - negativeCapPos ) * ( a.positiveSideRadius - a.negativeSideRadius ) + a.negativeSideRadius ) * lengthFac
        : a.positiveSideRadius; // Either radius is fine there, they should be equal at this point.

    // Signed distance from the sphere center to the conical surface (positive if outside).
    float signedDistToSurface = axisToSphereCenterDist * lengthFac - axisToSurfaceSlopedDist;

    // Signed distance from the sphere center to the positive cap, measured along the cap normal.
    float signedDistToPositiveCap = signedDistAlongAxis - a.positiveLength;
    // Signed distance from the sphere center to the negative cap, measured along the cap normal.
    float signedDistToNegativeCap = -a.negativeLength - signedDistAlongAxis;

    // Whether we're closer to the positive cap than the negative cap.
    bool positiveCapIsCloser = signedDistToPositiveCap > signedDistToNegativeCap;

    // Signed distance from the sphere center to the closest cap (positive if outside).
    float signedDistToClosestCap = positiveCapIsCloser ? signedDistToPositiveCap : signedDistToNegativeCap;

    MeasureResult ret;
    ret.distance.status = MeasureResult::Status::ok;

    bool haveDistance = false;

    if ( a.hollow || signedDistToSurface > signedDistToClosestCap )
    {
        if ( slopedSignedDistToPositiveCap <= 0 && slopedSignedDistToNegativeCap <= 0 )
        {
            // Near the conical surface.
            ret.distance.distance = ( a.hollow ? std::abs( signedDistToSurface ) : signedDistToSurface ) - b.radius;
            ret.distance.closestPointA = a.referencePoint + a.dir * projectedSpherePos + normalToConicalSurface * axisToSurfaceSlopedDist;
            ret.distance.closestPointB = b.center - normalToConicalSurface * b.radius * ( a.hollow && signedDistToSurface < 0 ? -1.f : 1.f );
            haveDistance = true;
        }
    }
    else
    {
        if ( signedDistToSurface <= 0 )
        {
            // Near the cap.
            ret.distance.distance = signedDistToClosestCap - b.radius;
            ret.distance.closestPointA = a.referencePoint + a.dir * ( positiveCapIsCloser ? a.positiveLength : -a.negativeLength ) + axisToSphereCenterDelta;
            ret.distance.closestPointB = b.center - a.dir * ( ( positiveCapIsCloser ? 1 : -1 ) * b.radius );
            haveDistance = true;
        }
    }

    // Near the edge.
    if ( !haveDistance )
    {
        // Distance from the sphere center to the cap edge, projected onto the normal to the cone axis.
        float distanceTowardsAxis = axisToSphereCenterDist - ( positiveCapIsSlopedCloser ? a.positiveSideRadius : a.negativeSideRadius );
        // Distance from the sphere center to the cap, projected onto the cone axis.
        float distanceAlongAxis = signedDistAlongAxis - ( positiveCapIsSlopedCloser ? a.positiveLength : -a.negativeLength );

        ret.distance.distance = std::sqrt( distanceAlongAxis * distanceAlongAxis + distanceTowardsAxis * distanceTowardsAxis ) - b.radius;
        ret.distance.closestPointA = a.referencePoint + a.dir * ( positiveCapIsSlopedCloser ? a.positiveLength : -a.negativeLength )
            + axisToSphereCenterDir * ( positiveCapIsSlopedCloser ? a.positiveSideRadius : a.negativeSideRadius );
        ret.distance.closestPointB = b.center - ( a.dir * distanceAlongAxis + axisToSphereCenterDir * distanceTowardsAxis ).normalized() * b.radius;

        haveDistance = true;
    }


    // Now the angle. (only for lines/segments <-> spheres, only if they collide)

    if ( !a.isZeroRadius() || b.radius == 0 )
    {
        ret.angle.status = MeasureResult::Status::badFeaturePair;
    }
    else if ( ret.distance.distance >= 0 )
    {
        // Line is outside the sphere.
        ret.angle.status = MeasureResult::Status::badRelativeLocation;
    }
    else if ( std::isfinite( a.positiveLength ) && std::isfinite( a.negativeLength ) &&
        ( a.basePoint( false ).center - b.center ).lengthSq() < b.radius * b.radius &&
        ( a.basePoint( true ).center - b.center ).lengthSq() < b.radius * b.radius )
    {
        // Line is inside the sphere.
        ret.angle.status = MeasureResult::Status::badRelativeLocation;
    }
    else
    {
        ret.angle.status = MeasureResult::Status::ok;

        Vector3f overlapMidpoint = b.center - axisToSphereCenterDir * axisToSphereCenterDist;
        float overlapHalfLen = std::sqrt( std::max( 0.f, b.radius * b.radius - axisToSphereCenterDist * axisToSphereCenterDist ) );

        bool backward = std::isfinite( a.positiveLength ) == std::isfinite( a.negativeLength )
            ? dot( a.centerPoint().center - b.center, a.dir ) < 0 : std::isfinite( a.positiveLength );

        ret.angle.pointA = ret.angle.pointB = overlapMidpoint + a.dir * ( overlapHalfLen * ( backward ? -1.f : 1.f ) );
        ret.angle.dirA = backward ? -a.dir : a.dir;
        ret.angle.dirB = ( ret.angle.pointB - b.center ).normalized();
        ret.angle.isSurfaceNormalA = false;
        ret.angle.isSurfaceNormalB = true;

        { // The intersection segment.
            Vector3f segmA = ret.angle.pointA;
            Vector3f segmB;

            if ( std::isfinite( a.positiveLength ) && ( a.basePoint( false ).center - b.center ).lengthSq() < b.radius * b.radius )
                segmB = a.basePoint( false ).center;
            else if ( std::isfinite( a.negativeLength ) && ( a.basePoint( true ).center - b.center ).lengthSq() < b.radius * b.radius )
                segmB = a.basePoint( true ).center;
            else
                segmB = overlapMidpoint + a.dir * ( overlapHalfLen * ( backward ? 1.f : -1.f ) );

            ret.intersections.push_back( toPrimitive( LineSegm3f( segmA, segmB ) ) );
        }
    }

    { // Center distance.
        if ( a.isZeroRadius() )
        {
            // If `a` is a line, just use the normal distance, but to the sphere center instead of the surface.
            ret.centerDistance = ret.distance;
            ret.centerDistance.distance += b.radius;
            ret.centerDistance.closestPointB = b.center;
        }
        else
        {
            // Otherwise just do the distance between centers.
            ret.centerDistance.status = MeasureResult::Status::ok;
            ret.centerDistance.closestPointA = a.centerPoint().center;
            ret.centerDistance.closestPointB = b.center;
            ret.centerDistance.distance = ( ret.centerDistance.closestPointB - ret.centerDistance.closestPointA ).length();
        }
    }

    return ret;
}

MeasureResult Binary<Primitives::Plane, Primitives::Sphere>::measure( const Primitives::Plane& a, const Primitives::Sphere& b ) const
{
    float signedCenterDist = dot( a.normal, b.center - a.center );

    // Distance.

    MeasureResult ret;
    ret.distance.status = MeasureResult::Status::ok;
    ret.distance.distance = std::abs( signedCenterDist ) - b.radius;
    ret.distance.closestPointA = b.center - a.normal * signedCenterDist;
    ret.distance.closestPointB = b.center - a.normal * ( b.radius * ( signedCenterDist >= 0 ? 1 : -1 ) );

    // Angle.

    if ( b.radius == 0 )
    {
        ret.angle.status = MeasureResult::Status::badFeaturePair;
    }
    else if ( ret.distance.distance > 0 )
    {
        ret.angle.status = MeasureResult::Status::badRelativeLocation;
    }
    else
    {
        float sideOffset = std::sqrt( std::max( 0.f, b.radius * b.radius - signedCenterDist * signedCenterDist ) );
        Vector3f sideDir = cross( a.normal, a.normal.furthestBasisVector() ).normalized();

        Vector3f intersectionCircleCenter = b.center - dot( a.normal, b.center - a.center ) * a.normal;

        ret.angle.status = MeasureResult::Status::ok;
        ret.angle.pointA = ret.angle.pointB = intersectionCircleCenter + sideDir * sideOffset;
        ret.angle.dirA = signedCenterDist > 0 ? a.normal : -a.normal;
        ret.angle.dirB = ( ret.angle.pointB - b.center ).normalized();
        ret.angle.isSurfaceNormalA = ret.angle.isSurfaceNormalB = true;

        ret.intersections.push_back( primitiveCircle( intersectionCircleCenter, ret.angle.dirA, sideOffset ) );
    }

    // Center distance.
    ret.centerDistance.status = MeasureResult::Status::ok;
    ret.centerDistance.distance = std::abs( signedCenterDist );
    ret.centerDistance.closestPointA = ret.distance.closestPointA;
    ret.centerDistance.closestPointB = b.center;

    return ret;
}

MeasureResult Binary<Primitives::ConeSegment, Primitives::ConeSegment>::measure( const Primitives::ConeSegment& a, const Primitives::ConeSegment& b ) const
{
    MeasureResult ret;

    // Distance.

    if ( a.isZeroRadius() && b.isZeroRadius() )
    {
        // https://math.stackexchange.com/a/4764188

        Vector3f nDenorm = cross( a.dir, b.dir );
        Vector3f n = nDenorm.normalized();
        Vector3f centerDelta = b.referencePoint - a.referencePoint;

        float signedDist = dot( n, centerDelta );

        Vector3f bCenterFixed = b.referencePoint - n * signedDist;

        float tFac = 1.f / nDenorm.lengthSq();

        float ta = dot( cross( bCenterFixed - a.referencePoint, b.dir ), nDenorm ) * tFac;
        float tb = dot( cross( bCenterFixed - a.referencePoint, a.dir ), nDenorm ) * tFac;

        ta = std::clamp( ta, -a.negativeLength, a.positiveLength );
        tb = std::clamp( tb, -b.negativeLength, b.positiveLength );

        ret.distance.status = MeasureResult::Status::ok;
        ret.distance.closestPointA = a.referencePoint + a.dir * ta;
        ret.distance.closestPointB = b.referencePoint + b.dir * tb;
        ret.distance.distance = ( ret.distance.closestPointB - ret.distance.closestPointA ).length();
    }
    else
    {
        // TODO: Support more cone types.
    }

    // Angle.

    auto isConeSuitableForAngle = [&] ( const Primitives::ConeSegment& cone )
    {
        return cone.positiveSideRadius == cone.negativeSideRadius;
    };
    if ( !isConeSuitableForAngle( a ) || !isConeSuitableForAngle( b ) )
    {
        ret.angle.status = MeasureResult::Status::badFeaturePair;
    }
    else
    {
        ret.angle.status = MeasureResult::Status::ok;
        if ( ret.distance )
        {
            ret.angle.pointA = ret.distance.closestPointA;
            ret.angle.pointB = ret.distance.closestPointB;
        }
        else
        {
            ret.angle.pointA = a.centerPoint().center;
            ret.angle.pointB = b.centerPoint().center;
        }

        auto guessDir = [&]( bool second ) -> Vector3f
        {
            const Primitives::ConeSegment& cone = second ? b : a;

            bool posFinite = std::isfinite( cone.positiveLength );
            bool negFinite = std::isfinite( cone.negativeLength );

            // If exactly one of the dirs is infinite, use it.
            if ( posFinite != negFinite )
                return negFinite ? cone.dir : -cone.dir;

            Vector3f center = cone.centerPoint().center;
            return dot( cone.dir, center - ret.angle.pointFor( second ) ) < 0 ? -cone.dir : cone.dir;
        };

        ret.angle.dirA = guessDir( false );
        ret.angle.dirB = guessDir( true );
        ret.angle.isSurfaceNormalA = ret.angle.isSurfaceNormalB = false;
    }

    // Center distance.

    // If at least one argument is a circle, replace it with the center point.
    if ( a.isCircle() && b.isCircle() )
    {
        ret.centerDistance = Features::measure( a.centerPoint(), b.centerPoint() ).centerDistance;
    }
    else if ( a.isCircle() )
    {
        ret.centerDistance = Features::measure( a.centerPoint(), b ).centerDistance;
    }
    else if ( b.isCircle() )
    {
        ret.centerDistance = Features::measure( a, b.centerPoint() ).centerDistance;
    }
    // Otherwise, if at least one argument has non-zero radius, set it to zero.
    else if ( !a.isZeroRadius() || !b.isZeroRadius() )
    {
        auto aLine = a;
        auto bLine = b;
        aLine.positiveSideRadius = aLine.negativeSideRadius = bLine.positiveSideRadius = bLine.negativeSideRadius = 0;
        ret.centerDistance = measure( aLine, bLine ).centerDistance;
    }
    else
    {
        // Otherwise, if the angle between lines is >45 degrees, calculate the distance as usual.
        float dirDot = dot( a.dir, b.dir );
        if ( std::abs( dirDot ) < 0.707106781186547524401f ) // sqrt(2)/2, aka M_SQRT1_2
        {
            ret.centerDistance = ret.distance;
        }
        // If one of the lines is finite and the other isn't, calculate distance from the center of the finite line to the entire other line.
        else if ( std::isfinite( a.length() ) != std::isfinite( b.length() ) )
        {
            if ( std::isfinite( a.length() ) )
                ret.centerDistance = Features::measure( a.centerPoint(), b ).distance;
            else
                ret.centerDistance = Features::measure( a, b.centerPoint() ).distance;
        }
        // Otherwise use a weird approximation that assumes that the lines are mostly parallel.
        else
        {
            Vector3f bDirFixed = b.dir;
            if ( dirDot < 0 )
                bDirFixed = -bDirFixed;
            Vector3f averageDir = ( a.dir + bDirFixed ).normalized();

            Vector3f aCenter = a.centerPoint().center;
            Vector3f bCenter = b.centerPoint().center;

            float offset = dot( bCenter - aCenter, averageDir ) / dot( averageDir, a.dir );

            ret.centerDistance.status = MeasureResult::Status::ok;
            ret.centerDistance.closestPointA = aCenter + a.dir * offset;
            ret.centerDistance.closestPointB = bCenter + bDirFixed * offset;
            ret.centerDistance.distance = ( ret.centerDistance.closestPointB - ret.centerDistance.closestPointA ).length();
        }
    }

    return ret;
}

MeasureResult Binary<Primitives::Plane, Primitives::ConeSegment>::measure( const Primitives::Plane& a, const Primitives::ConeSegment& b ) const
{
    MeasureResult ret;

    bool havePositivePoints = false, haveNegativePoints = false;

    if ( !std::isfinite( b.positiveLength ) && !std::isfinite( b.negativeLength ) )
    {
        ret.distance.status = MeasureResult::Status::badFeaturePair;
    }
    else
    {
        // A normal to the cone axis, parallel to the plane normal. The sign of this is unspecified.
        Vector3f sideDir = cross( cross( a.normal, b.dir ), b.dir ).normalized();
        if ( sideDir == Vector3f() || !sideDir.isFinite() )
            sideDir = cross( b.dir, b.dir.furthestBasisVector() ).normalized(); // An arbitrary direction.

        Vector3f positiveCapCenter = b.referencePoint + b.dir * b.positiveLength;
        Vector3f negativeCapCenter = b.referencePoint - b.dir * b.negativeLength;

        bool first = true;

        float maxDist = 0, minDist = 0;
        Vector3f maxDistPoint, minDistPoint;

        for ( bool positiveSide : { true, false } )
        {
            if ( !std::isfinite( positiveSide ? b.positiveLength : b.negativeLength ) )
            {
                float dirDot = dot( a.normal, b.dir * ( positiveSide ? 1.f : -1.f ) );

                float dist = 0;
                if ( std::abs( dirDot ) < 0.00001f ) // TODO move the epsilon to a constant?
                {
                    // Shrug. This fixes an edge case, but I'm not sure if I should even bother with this.
                    continue;
                }
                else if ( dirDot < 0 )
                {
                    haveNegativePoints = true;
                    dist = -INFINITY;
                }
                else
                {
                    havePositivePoints = true;
                    dist = INFINITY;
                }

                if ( first || dist < minDist )
                    minDist = dist;
                if ( first || dist > maxDist )
                    maxDist = dist;

                first = false;

                continue;
            }

            Vector3f capCenter = positiveSide ? positiveCapCenter : negativeCapCenter;
            float sideRadius = positiveSide ? b.positiveSideRadius : b.negativeSideRadius;

            for ( Vector3f point : {
                capCenter + sideDir * sideRadius,
                capCenter - sideDir * sideRadius,
            } )
            {
                float dist = dot( a.normal, point - a.center );
                ( dist < 0 ? haveNegativePoints : havePositivePoints ) = true;

                if ( first || dist < minDist )
                {
                    minDist = dist;
                    minDistPoint = point;
                }
                if ( first || dist > maxDist )
                {
                    maxDist = dist;
                    maxDistPoint = point;
                }

                first = false;
            }
        }

        assert( havePositivePoints || haveNegativePoints );

        ret.distance.status = MeasureResult::Status::ok;

        if ( !havePositivePoints || ( haveNegativePoints && maxDist < -minDist ) )
        {
            ret.distance.distance = maxDist;
            ret.distance.closestPointB = maxDistPoint;
        }
        else
        {
            ret.distance.distance = minDist;
            ret.distance.closestPointB = minDistPoint;
        }

        ret.distance.distance = std::abs( ret.distance.distance );
        if ( havePositivePoints && haveNegativePoints )
            ret.distance.distance = -ret.distance.distance;

        ret.distance.closestPointA = ret.distance.closestPointB - a.normal * dot( a.normal, ret.distance.closestPointB - a.center );
    }


    // Angle.

    if ( !b.isCircle() && !b.isZeroRadius() &&
        ( b.positiveSideRadius != b.negativeSideRadius || std::isfinite( b.positiveLength ) || std::isfinite( b.negativeLength ) ) )
    {
        ret.angle.status = MeasureResult::Status::badFeaturePair;
    }
    else
    {
        ret.angle.status = MeasureResult::Status::ok;
        ret.angle.dirA = a.normal; // This should be already normalized.
        ret.angle.dirB = b.dir; // This should also be already normalized.
        ret.angle.isSurfaceNormalA = true;
        ret.angle.isSurfaceNormalB = b.isCircle();

        // If the absolute value of `cos()` is smaller than this, we don't try to look for the intersection point, since it can be far away.
        float cosThreshold = 0.008f; // 90 +- ~0.5 degrees
        if ( b.isCircle()
            ? cross( ret.angle.dirA, ret.angle.dirB ).lengthSq() < cosThreshold * cosThreshold
            : std::abs( dot( ret.angle.dirA, ret.angle.dirB ) ) < cosThreshold
        )
        {
            if ( ret.distance )
            {
                ret.angle.pointA = ret.distance.closestPointA;
                ret.angle.pointB = ret.distance.closestPointB;
            }
            else
            {
                ret.angle.pointA = a.center;
                ret.angle.pointB = b.centerPoint().center;
            }
        }
        else if ( b.isCircle() )
        {
            auto intersectionLine = b.basePlane( false ).intersectWithPlane( a );
            ret.angle.pointA = ret.angle.pointB = intersectionLine.referencePoint;
            ret.intersections.push_back( std::move( intersectionLine ) );
        }
        else
        {
            auto intersectionPoint = a.intersectWithLine( b );
            ret.angle.pointA = ret.angle.pointB = intersectionPoint.center;
            ret.intersections.push_back( std::move( intersectionPoint ) );
        }
    }

    { // Center distance.
        // Just the distance to B's center, nothing fancy.

        Vector3f bCenter = b.centerPoint().center;

        float signedCenterDist = dot( a.normal, bCenter - a.center );

        ret.centerDistance.status = MeasureResult::Status::ok;
        ret.centerDistance.distance = std::abs( signedCenterDist );
        ret.centerDistance.closestPointA = bCenter - a.normal * signedCenterDist;
        ret.centerDistance.closestPointB = bCenter;
    }

    return ret;
}

MeasureResult Binary<Primitives::Plane, Primitives::Plane>::measure( const Primitives::Plane& a, const Primitives::Plane& b ) const
{
    MeasureResult ret;

    // We're not going to check for parallel-ness with some epsilon.
    // You can just pick a point on one of the planes instead, or use center-distance.
    ret.distance.status = MeasureResult::Status::badFeaturePair;

    // Angle.

    auto intersectionLine = a.intersectWithPlane( b );

    ret.angle.status = MeasureResult::Status::ok;
    ret.angle.pointA = ret.angle.pointB = intersectionLine.referencePoint;
    ret.angle.dirA = a.normal;
    ret.angle.dirB = b.normal;
    ret.angle.isSurfaceNormalA = ret.angle.isSurfaceNormalB = true;

    if ( std::abs( dot( a.normal, b.normal ) ) < 0.99995f ) // ~0.5 degrees
        ret.intersections.push_back( std::move( intersectionLine ) );

    { // Center distance.
        Vector3f bNormalFixed = b.normal;
        if ( dot( a.normal, b.normal ) < 0 )
            bNormalFixed = -bNormalFixed;
        Vector3f averageNormal = ( a.normal + bNormalFixed ).normalized();

        Vector3f bCenterProjected = b.center - averageNormal * dot( averageNormal, b.center - a.center );

        Vector3f averageCenter = ( bCenterProjected - a.center ) * 0.5f + a.center;

        auto connectingLine = toPrimitive( Line3f( averageCenter, averageNormal ) );

        ret.centerDistance.status = MeasureResult::Status::ok;
        ret.centerDistance.closestPointA = a.intersectWithLine( connectingLine ).center;
        ret.centerDistance.closestPointB = b.intersectWithLine( connectingLine ).center;
        ret.centerDistance.distance = ( ret.centerDistance.closestPointB - ret.centerDistance.closestPointA ).length();
    }

    return ret;
}

} // namespace Traits

std::string name( const Primitives::Variant& var )
{
    return std::visit( []( const auto& elem ){ return (name)( elem ); }, var );
}
MeasureResult measure( const Primitives::Variant& a, const Primitives::Variant& b )
{
    return std::visit( []( const auto& elemA, const auto& elemB ){ return (measure)( elemA, elemB ); }, a, b );
}


static constexpr float testEps = 0.0001f;

TEST( Features, PrimitiveConstruction )
{
    { // Infinite line to cone segment.
        Vector3f pos( 10, 20, 35 );
        Vector3f dir( 0, -3, 0 );
        auto cone = toPrimitive( Line3f( pos, dir ) );

        ASSERT_EQ( cone.positiveSideRadius, 0 );
        ASSERT_EQ( cone.negativeSideRadius, 0 );
        ASSERT_LE( ( cone.referencePoint - pos ).length(), testEps );
        ASSERT_LE( ( cone.dir - Vector3f( 0, -1, 0 ) ).length(), testEps );
        ASSERT_EQ( cone.positiveLength, INFINITY );
        ASSERT_EQ( cone.negativeLength, INFINITY );
    }

    { // Finite line to cone segment.
        Vector3f pos( 10, 20, 35 );
        Vector3f dir( 0, -3, 0 );
        auto cone = toPrimitive( LineSegm3f( pos, pos + dir ) );

        ASSERT_EQ( cone.positiveSideRadius, 0 );
        ASSERT_EQ( cone.negativeSideRadius, 0 );
        ASSERT_LE( ( cone.referencePoint - pos ).length(), testEps );
        ASSERT_LE( ( cone.dir - Vector3f( 0, -1, 0 ) ).length(), testEps );
        ASSERT_NEAR( cone.positiveLength, 3, testEps );
        ASSERT_NEAR( cone.negativeLength, 0, testEps );
    }

    { // Cone segment defined as a cylinder.
        Vector3f pos( 10, 20, 35 );
        Vector3f dir( 0, -3, 0 );
        float rad = 4;
        auto cone = primitiveCylinder( pos, pos + dir, rad );
        ASSERT_EQ( cone.positiveSideRadius, rad );
        ASSERT_EQ( cone.negativeSideRadius, rad );
        ASSERT_LE( ( cone.referencePoint - pos ).length(), testEps );
        ASSERT_LE( ( cone.dir - Vector3f( 0, -1, 0 ) ).length(), testEps );
        ASSERT_NEAR( cone.positiveLength, 3, testEps );
        ASSERT_NEAR( cone.negativeLength, 0, testEps );
    }

    { // Cone segment defined as a cone.
        Vector3f pos( 10, 20, 35 );
        Vector3f dir( 0, -3, 0 );
        float rad = 4;
        auto cone = primitiveCylinder( pos, pos + dir, rad );
        ASSERT_EQ( cone.positiveSideRadius, rad );
        ASSERT_EQ( cone.negativeSideRadius, rad );
        ASSERT_LE( ( cone.referencePoint - pos ).length(), testEps );
        ASSERT_LE( ( cone.dir - Vector3f( 0, -1, 0 ) ).length(), testEps );
        ASSERT_NEAR( cone.positiveLength, 3, testEps );
        ASSERT_NEAR( cone.negativeLength, 0, testEps );
    }
}

TEST( Features, PrimitiveOps_ConeSegment )
{
    { // Untruncate.
        Primitives::ConeSegment cone{
            .referencePoint = Vector3f( 100, 50, 10 ),
            .dir = Vector3f( 1, 0, 0 ),
            .positiveSideRadius = 10,
            .negativeSideRadius = 15,
            .positiveLength = 2,
            .negativeLength = 4,
        };

        ASSERT_EQ( cone.untruncateCone().positiveLength, 14 );
        std::swap( cone.positiveSideRadius, cone.negativeSideRadius );
        ASSERT_EQ( cone.untruncateCone().negativeLength, 16 );
    }
}

TEST( Features, Distance_Sphere_Sphere )
{
    { // Point-point.
        { // Overlap.
            Vector3f a( 10, 20, 30 );
            auto r = measure( toPrimitive( a ), toPrimitive( a ) ).distance;
            ASSERT_NEAR( r.distance, 0, testEps );
            ASSERT_LE( ( r.closestPointA - a ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - a ).length(), testEps );
        }

        { // Positive distance.
            Vector3f a( 10, 20, 30 ), b( 7, 3, 1 );
            auto r = measure( toPrimitive( a ), toPrimitive( b ) ).distance;
            ASSERT_NEAR( r.distance, ( b - a ).length(), testEps );
            ASSERT_LE( ( r.closestPointA - a ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - b ).length(), testEps );
        }
    }

    // Not checking sphere-point, that should just work.

    { // Sphere-sphere.
        const Primitives::Sphere sphere( Vector3f( 10, 20, 30 ), 7 );

        { // Center overlap.
            auto sphere2 = sphere;
            sphere2.radius = 4;

            Vector3f arbitraryDir( 1, 0, 0 ); // This is hardcoded in the algorithm, and is used for ambiguities.

            auto r = measure( sphere, sphere2 ).distance;
            ASSERT_NEAR( r.distance, -( sphere.radius + sphere2.radius ), testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + arbitraryDir * sphere.radius ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - arbitraryDir * sphere2.radius ) ).length(), testEps );
        }

        { // Negative distance.
            auto sphere2 = sphere;
            sphere2.radius = 4;
            float xOffset = 5;
            sphere2.center.x += xOffset;

            auto r = measure( sphere, sphere2 ).distance;
            ASSERT_NEAR( r.distance, xOffset - sphere.radius - sphere2.radius, testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + Vector3f( sphere.radius, 0, 0 ) ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - Vector3f( sphere2.radius, 0, 0 ) ) ).length(), testEps );
        }

        { // Positive distance.
            auto sphere2 = sphere;
            sphere2.radius = 4;
            float xOffset = 20;
            sphere2.center.x += xOffset;

            auto r = measure( sphere, sphere2 ).distance;
            ASSERT_NEAR( r.distance, xOffset - sphere.radius - sphere2.radius, testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + Vector3f( sphere.radius, 0, 0 ) ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - Vector3f( sphere2.radius, 0, 0 ) ) ).length(), testEps );
        }
    }
}

TEST( Features, Distance_ConeSegment_Sphere )
{
    { // Line to sphere.
        for ( bool lineIsInfinite : { true, false } )
        {
            Vector3f linePos( 10, 30, 70 );
            Vector3f lineDir( 0, -1, 0 ); // Must be normalized.
            float lineStep = 3.5f;
            Primitives::ConeSegment line;
            if ( lineIsInfinite )
            {
                line = toPrimitive( Line3f( linePos, lineDir ) );
            }
            else
            {
                // Not using `LineSegm3f` to have non-trivial positive and negative lengths.
                line = Primitives::ConeSegment{ .referencePoint = linePos, .dir = lineDir, .positiveLength = lineStep * 2, .negativeLength = lineStep };
            }

            for ( float sphereRad : { 0.f, 2.f, 10.f } )
            {
                for ( float fac : { -1.2f, -1.f, -0.8f, 0.f, 0.2f, 1.8f, 2.f, 2.2f } )
                {
                    // If the line is finite, this is the non-negative distance to it projected onto the line itself.
                    float distToCap = lineIsInfinite ? 0 : std::max( { 0.f, fac - 2, -1 - fac } ) * lineStep;

                    Vector3f closestPointOnLine;
                    if ( lineIsInfinite )
                        closestPointOnLine = linePos + lineDir * lineStep * fac;
                    else
                        closestPointOnLine = linePos + lineDir * std::clamp( fac * lineStep, -line.negativeLength, line.positiveLength );

                    { // Center overlaps the line.
                        Primitives::Sphere sphere( linePos + lineDir * lineStep * fac, sphereRad );
                        auto r = measure( line, sphere ).distance;
                        ASSERT_NEAR( r.distance, distToCap - sphereRad, testEps );
                        ASSERT_LE( ( r.closestPointA - closestPointOnLine ).length(), testEps );
                        ASSERT_NEAR( ( r.closestPointB - sphere.center ).length(), sphere.radius, testEps ); // Arbitrary direction here, due to an ambiguity.
                    }

                    { // Center doesn't overlap the line.
                        Vector3f offset( 1, 0, 3 );
                        Primitives::Sphere sphere( linePos + lineDir * lineStep * fac + offset, sphereRad );
                        auto r = measure( line, sphere ).distance;
                        float expectedDist = std::sqrt( offset.lengthSq() + distToCap * distToCap ) - sphereRad;
                        ASSERT_NEAR( r.distance, expectedDist, testEps );
                        ASSERT_LE( ( r.closestPointA - closestPointOnLine ).length(), testEps );
                        ASSERT_LE( ( r.closestPointB - ( sphere.center + ( closestPointOnLine - sphere.center ).normalized() * sphere.radius ) ).length(), testEps );
                    }
                }
            }
        }
    }

    { // Cone to sphere.
        for ( bool hollowCone : { false, true } )
        {
            Primitives::ConeSegment cone{
                .referencePoint = Vector3f( 100, 200, 50 ),
                .dir = Vector3f( 1, 0, 0 ),
                .positiveSideRadius = 8,
                .negativeSideRadius = 14,
                .positiveLength = 20,
                .negativeLength = 10,
                .hollow = hollowCone,
            };

            for ( bool invertCone : { false, true } )
            {
                if ( invertCone )
                {
                    cone.dir = -cone.dir;
                    std::swap( cone.positiveLength, cone.negativeLength );
                    std::swap( cone.positiveSideRadius, cone.negativeSideRadius );
                }

                auto testPoint = [&]( Vector3f pos, float expectedDist, Vector3f expectedPointOnCone )
                {
                    pos += cone.referencePoint;
                    expectedPointOnCone += cone.referencePoint;
                    Primitives::Sphere sphere( pos, 3 );
                    auto r = measure( cone, sphere ).distance;
                    ASSERT_NEAR( r.distance, expectedDist, testEps );
                    ASSERT_LE( ( r.closestPointA - expectedPointOnCone ).length(), testEps );

                    if ( expectedPointOnCone == sphere.center )
                    {
                        ASSERT_NEAR( ( r.closestPointB - sphere.center ).length(), sphere.radius, testEps );
                    }
                    else
                    {
                        Vector3f spherePointOffset = ( expectedPointOnCone - sphere.center ).normalized() * sphere.radius * ( expectedDist < -sphere.radius ? -1.f : 1.f );
                        ASSERT_LE( ( r.closestPointB - ( sphere.center + spherePointOffset ) ).length(), testEps );
                    }
                };

                // Flat caps.
                if ( !hollowCone )
                {
                    for ( float y : { 0.f, 4.f } )
                    {
                        // Positive cap.
                        testPoint( Vector3f( 24, y, 0 ), 1, Vector3f( 20, y, 0 ) );
                        testPoint( Vector3f( 23, y, 0 ), 0, Vector3f( 20, y, 0 ) );
                        testPoint( Vector3f( 22, y, 0 ), -1, Vector3f( 20, y, 0 ) );
                        testPoint( Vector3f( 20, y, 0 ), -3, Vector3f( 20, y, 0 ) );
                        testPoint( Vector3f( 19, y, 0 ), -4, Vector3f( 20, y, 0 ) );
                        testPoint( Vector3f( 16, y, 0 ), -7, Vector3f( 20, y, 0 ) );

                        // Negative cap.
                        testPoint( Vector3f( -14, y, 0 ), 1, Vector3f( -10, y, 0 ) );
                        testPoint( Vector3f( -13, y, 0 ), 0, Vector3f( -10, y, 0 ) );
                        testPoint( Vector3f( -12, y, 0 ), -1, Vector3f( -10, y, 0 ) );
                        testPoint( Vector3f( -10, y, 0 ), -3, Vector3f( -10, y, 0 ) );
                        testPoint( Vector3f( -9, y, 0 ), -4, Vector3f( -10, y, 0 ) );
                        testPoint( Vector3f( -6, y, 0 ), -7, Vector3f( -10, y, 0 ) );
                    }
                }

                // Cap edges.
                for ( bool positive : { true, false } )
                {
                    Vector3f point = positive ? Vector3f( 20, 8, 0 ) : Vector3f( -10, 14, 0 );

                    Vector3f capNormal( positive ? 1.f : -1.f, 0, 0 );
                    Vector3f coneNormal = Vector3f( 6, 30, 0 ).normalized();
                    Vector3f dirToTip = Vector3f( 30, -6, 0 ).normalized() * ( positive ? 1.f : -1.f );
                    Vector3f averageNormal = ( coneNormal + capNormal ).normalized();

                    // The edges themselves.
                    for ( Vector3f dir : {
                        capNormal,
                        ( capNormal + averageNormal ).normalized(),
                        averageNormal,
                        ( coneNormal + averageNormal ).normalized(),
                        coneNormal,
                    } )
                    {
                        for ( float dist : { 0.f, 2.f, 3.f, 4.f } )
                            testPoint( point + dir * dist, dist - 3, point );
                    }

                    // Internal cutoff point: cap to conical surface.
                    if ( !hollowCone )
                    {
                        float cutoffOffset = 0.001f;

                        Vector3f cutoffPoint( point.x - point.y * averageNormal.x / averageNormal.y, 0.00001f, 0 );

                        // Closer to the cap.
                        auto a = measure( cone, Primitives::Sphere( cone.referencePoint + cutoffPoint + capNormal * cutoffOffset, 1 ) ).distance;
                        ASSERT_LE( ( a.closestPointA - ( cone.referencePoint + Vector3f( point.x, 0, 0 ) ) ).length(), testEps );

                        // Closer to the surface.
                        auto b = measure( cone, Primitives::Sphere( cone.referencePoint + cutoffPoint - capNormal * cutoffOffset, 1 ) ).distance;
                        Vector3f expectedPoint = cone.referencePoint + point - dirToTip * point.y;
                        ASSERT_LE( ( b.closestPointA - expectedPoint ).length(), 0.001f );
                    }
                }

                { // Near the conical surface.
                    Vector3f offset = Vector3f( 6, 30, 0 ).normalized();
                    for ( float dist : { 5.f, -7.5f } )
                    {
                        if ( dist < 0 && !cone.hollow )
                            continue;

                        Vector3f a = Vector3f( 20, 8, 0 ) + offset * dist;
                        Vector3f b = Vector3f( -10, 14, 0 ) + offset * dist;

                        for ( float t : { 0.f, 1.f, 0.5f, 0.2f, 0.8f } )
                        {
                            Primitives::Sphere sphere( cone.referencePoint + a * ( 1 - t ) + b * t, 1 );
                            auto r = measure( cone, sphere ).distance;
                            ASSERT_NEAR( r.distance, std::abs( dist ) - sphere.radius, testEps );
                            ASSERT_LE( ( r.closestPointA - ( sphere.center - offset * dist ) ).length(), testEps );
                            ASSERT_LE( ( r.closestPointB - ( sphere.center - offset * sphere.radius * ( dist < 0 ? -1.f : 1.f ) ) ).length(), testEps );
                        }
                    }
                }
            }
        }
    }
}

TEST( Features, Distance_Plane_Sphere )
{
    Vector3f planeCenter = Vector3f( 100, 50, 7 );
    Primitives::Plane plane{ .center = planeCenter, .normal = Vector3f( 1, 0, 0 ) };
    Vector3f sideOffset( 0, -13, 71 );

    for ( float dist : { -4.f, -2.f, 0.f, 2.f, 4.f } )
    {
        Primitives::Sphere sphere( planeCenter + sideOffset + plane.normal * dist, 3.f );
        auto r = measure( plane, sphere ).distance;

        ASSERT_NEAR( r.distance, std::abs( dist ) - sphere.radius, testEps );
        ASSERT_LE( ( r.closestPointA - ( planeCenter + sideOffset ) ).length(), testEps );

        if ( dist == 0.f )
        {
            ASSERT_TRUE(
                ( r.closestPointB - ( sphere.center + plane.normal * sphere.radius ) ).length() < testEps ||
                ( r.closestPointB - ( sphere.center - plane.normal * sphere.radius ) ).length() < testEps
            );
        }
        else
        {
            ASSERT_LE( ( r.closestPointB - ( sphere.center - plane.normal * sphere.radius * ( dist > 0 ? 1.f : -1.f ) ) ).length(), testEps );
        }
    }
}

TEST( Features, Distance_ConeSegment_ConeSegment )
{
    { // Line-line.
        { // Infinite lines.
            { // Skew lines (not parallel and not intersecting).
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 10 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 ).normalized();

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = measure( l1, l2 ).distance;
                ASSERT_NEAR( r.distance, 10, testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 102, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - Vector3f( 102, 50, 20 ) ).length(), testEps );
            }

            { // An exact intersection.
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 0 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 ).normalized();

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = measure( l1, l2 ).distance;
                ASSERT_LE( r.distance, testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 102, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - r.closestPointA ).length(), testEps );
            }

            { // Parallel.
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 0 ), d1( 1, 0, 0 ), d2 = d1;

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = measure( l1, l2 ).distance;
                ASSERT_EQ( r.status, MeasureResult::Status::badRelativeLocation );
            }
        }

        { // Finite lines.
            { // Skew lines (not parallel and not intersecting).
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 2, 5 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 );

                auto l1 = toPrimitive( LineSegm3f( p1, p1 + d1 ) );
                auto l2 = toPrimitive( LineSegm3f( p2 + d2, p2 ) ); // Backwards, why not.

                auto r = measure( l1, l2 ).distance;
                ASSERT_NEAR( r.distance, std::sqrt( 1 + 1 + 5*5 ), testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 101, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - Vector3f( 102, 51, 15 ) ).length(), testEps );
            }
        }
    }
}

TEST( Features, Distance_Plane_ConeSegment )
{
    auto testPlane = [&]( Primitives::ConeSegment originalCone, Vector3f surfacePoint, Vector3f offsetIntoCone,
        // Our surface points may or may not be offset by one of those vectors, if specified:
        Vector3f surfacePointSlideA = {}, Vector3f surfacePointSlideB = {},
        bool distIsAbs = false
    )
    {
        auto checkCone = [&]( const Primitives::ConeSegment& cone )
        {
            for ( float fac : { 0.f, 1.f, -2.f } )
            {
                Vector3f closestPlanePoint = surfacePoint + offsetIntoCone * fac;

                auto checkPlane = [&]( const Primitives::Plane& plane )
                {
                    float expectedDist = -fac * offsetIntoCone.length();
                    if ( distIsAbs )
                        expectedDist = std::abs( expectedDist );

                    auto r = measure( cone, plane ).distance;
                    ASSERT_NEAR( r.distance, expectedDist, testEps );

                    Vector3f slide;
                    ASSERT_TRUE(
                        ( r.closestPointA - surfacePoint ).length() < testEps ||
                        ( r.closestPointA - surfacePoint - ( slide = surfacePointSlideA ) ).length() < testEps ||
                        ( r.closestPointA - surfacePoint - ( slide = surfacePointSlideB ) ).length() < testEps
                    );
                    ASSERT_LE( ( r.closestPointB - closestPlanePoint - slide ).length(), testEps );
                };

                Vector3f randomPlaneCenterSlide = cross( offsetIntoCone, offsetIntoCone.furthestBasisVector() ).normalized() * 42.f;
                Primitives::Plane plane{ .center = closestPlanePoint + randomPlaneCenterSlide, .normal = offsetIntoCone.normalized() };

                checkPlane( plane );
                plane.normal = -plane.normal;
                checkPlane( plane );
            }
        };

        checkCone( originalCone );

        originalCone.dir = -originalCone.dir;
        std::swap( originalCone.positiveLength, originalCone.negativeLength );
        std::swap( originalCone.positiveSideRadius, originalCone.negativeSideRadius );
        checkCone( originalCone );
    };

    { // Finite cone.
        Primitives::ConeSegment cone{
            .referencePoint = Vector3f( 100, 50, 10 ),
            .dir = Vector3f( 1, 0, 0 ),
            .positiveSideRadius = 8,
            .negativeSideRadius = 14,
            .positiveLength = 20,
            .negativeLength = 10,
        };

        testPlane( cone, cone.referencePoint + Vector3f( 20, 0, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, 8, 0 ), Vector3f( 0, -8, 0 ) );

        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, -16, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( -1, -1, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( -1, -2, 0 ) );

        testPlane( cone, cone.referencePoint + Vector3f( -10, 14, 0 ), Vector3f( 1, -2, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( -10, 14, 0 ), Vector3f( 1, -1, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( -10, 14, 0 ), Vector3f( 1, 0, 0 ), Vector3f( 0, -28, 0 ) );

        testPlane( cone, cone.referencePoint + Vector3f( -10, 0, 0 ), Vector3f( 1, 0, 0 ), Vector3f( 0, 14, 0 ), Vector3f( 0, -14, 0 ) );
    }

    { // Half-infinite cone.
        Primitives::ConeSegment cone{
            .referencePoint = Vector3f( 100, 50, 10 ),
            .dir = Vector3f( 1, 0, 0 ),
            .positiveSideRadius = 8,
            .negativeSideRadius = 8,
            .positiveLength = 20,
            .negativeLength = INFINITY,
        };

        testPlane( cone, cone.referencePoint + Vector3f( 20, 0, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, 8, 0 ), Vector3f( 0, -8, 0 ) );

        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( -1, -1, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( -100, -200, 0 ) );

        testPlane( cone, cone.referencePoint + Vector3f( 20, 8, 0 ), Vector3f( 0, -1, 0 ) );
        testPlane( cone, cone.referencePoint + Vector3f( -40, 8, 0 ), Vector3f( 0, -1, 0 ), Vector3f( 60, 0, 0 ) );
    }

    { // A line.
        { // Infinite.
            Primitives::ConeSegment line{
                .referencePoint = Vector3f( 100, 50, 10 ),
                .dir = Vector3f( 1, 0, 0 ),
                .positiveSideRadius = 0,
                .negativeSideRadius = 0,
                .positiveLength = INFINITY,
                .negativeLength = INFINITY,
            };

            Primitives::Plane plane{ .center = Vector3f( 1, 2, 3 ), .normal = Vector3f( 5, 6, 7 ).normalized() };

            auto r = measure( line, plane ).distance;
            ASSERT_EQ( r.status, MeasureResult::Status::badFeaturePair );
        }

        { // Finite.
            Primitives::ConeSegment line{
                .referencePoint = Vector3f( 100, 50, 10 ),
                .dir = Vector3f( 1, 0, 0 ),
                .positiveSideRadius = 0,
                .negativeSideRadius = 0,
                .positiveLength = 20,
                .negativeLength = 10,
            };

            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -1, 0, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -1, -1, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -4, -3, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( 0, -1, 0 ), Vector3f( -30, 0, 0 ), {}, true );

            testPlane( line, Vector3f( 100, 50, 10 ), Vector3f( 0, -1, 0 ), Vector3f( 20, 0, 0 ), Vector3f( -10, 0, 0 ), true );

            testPlane( line, Vector3f( 90, 50, 10 ), Vector3f( 1, 0, 0 ) );
            testPlane( line, Vector3f( 90, 50, 10 ), Vector3f( 1, -1, 0 ) );
            testPlane( line, Vector3f( 90, 50, 10 ), Vector3f( 4, -3, 0 ) );
            testPlane( line, Vector3f( 90, 50, 10 ), Vector3f( 0, -1, 0 ), Vector3f( 30, 0, 0 ), {}, true );
        }

        { // Half-finite.
            Primitives::ConeSegment line{
                .referencePoint = Vector3f( 100, 50, 10 ),
                .dir = Vector3f( 1, 0, 0 ),
                .positiveSideRadius = 0,
                .negativeSideRadius = 0,
                .positiveLength = 20,
                .negativeLength = INFINITY,
            };

            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -1, 0, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -1, -1, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( -100, -200, 0 ) );
            testPlane( line, Vector3f( 120, 50, 10 ), Vector3f( 0, -1, 0 ), {}, {}, true );

            testPlane( line, Vector3f( 100, 50, 10 ), Vector3f( 0, -1, 0 ), Vector3f( 20, 0, 0 ), {}, true );
            testPlane( line, Vector3f( 30, 50, 10 ), Vector3f( 0, -1, 0 ), Vector3f( 90, 0, 0 ), {}, true );
        }
    }
}

TEST( Features, Angle_Sphere_Sphere )
{
    { // Normal.
        Primitives::Sphere a( Vector3f( 100, 50, 10 ), 5 );
        Primitives::Sphere b( Vector3f( 107, 50, 10 ), 3 * std::sqrt( 2.f ) );

        auto r = measure( a, b );
        ASSERT_EQ( r.angle.status, MeasureResult::Status::ok );

        ASSERT_EQ( r.angle.pointA, r.angle.pointB );

        ASSERT_NEAR( r.angle.pointA.x, 104, testEps );
        ASSERT_NEAR( ( r.angle.pointA - Vector3f( 104, 50, 10 ) ).length(), 3, testEps );

        ASSERT_LE( ( r.angle.dirA - ( r.angle.pointA - a.center ).normalized() ).length(), testEps );
        ASSERT_LE( ( r.angle.dirB - ( r.angle.pointB - b.center ).normalized() ).length(), testEps );

        ASSERT_TRUE( r.angle.isSurfaceNormalA );
        ASSERT_TRUE( r.angle.isSurfaceNormalB );
    }

    { // Spheres not colliding.
        { // Outside.
            Primitives::Sphere a( Vector3f( 100, 50, 10 ), 5 );
            Primitives::Sphere b( Vector3f( 108.01f, 50, 10 ), 3 );

            auto r = measure( a, b );
            ASSERT_EQ( r.angle.status, MeasureResult::Status::badRelativeLocation );
        }

        { // Inside.
            Primitives::Sphere a( Vector3f( 100, 50, 10 ), 5 );
            Primitives::Sphere b( Vector3f( 101.99f, 50, 10 ), 3 );

            auto r = measure( a, b );
            ASSERT_EQ( r.angle.status, MeasureResult::Status::badRelativeLocation );
        }
    }

    { // One of the spheres is a point.
        Primitives::Sphere a( Vector3f( 100, 50, 10 ), 5 );
        Primitives::Sphere b( Vector3f( 108.01f, 50, 10 ), 0 );

        auto r = measure( a, b );
        ASSERT_EQ( r.angle.status, MeasureResult::Status::badFeaturePair );
    }
}

TEST( Features, Angle_ConeSegment_Sphere )
{
    { // Normal.
        Primitives::Sphere a( Vector3f( 100, 50, 10 ), 8 );

        { // To infinite line.
            for ( bool backward : { false, true } )
            {
                auto b = toPrimitive( Line3f( Vector3f( 100.f + ( backward ? -1.f : 1.f ), 54, 10 ), Vector3f( 1, 0, 0 ) ) );

                auto r = measure( a, b );

                ASSERT_EQ( r.angle.status, MeasureResult::Status::ok );

                ASSERT_EQ( r.angle.pointA, r.angle.pointB );
                Vector3f dir( std::sin( MR::PI_F / 3 ) * ( backward ? -1.f : 1.f ), 0.5f, 0 );
                Vector3f expectedPointA = a.center + Vector3f( dir * a.radius );
                ASSERT_LE( ( r.angle.pointA - expectedPointA ).length(), testEps );

                ASSERT_LE( ( r.angle.dirA - dir ).length(), testEps );
                // `dir` being conditionally flipped here is a bit arbitrary, but looks good to me.
                ASSERT_LE( ( r.angle.dirB - b.dir * ( backward ? -1.f : 1.f ) ).length(), testEps );

                ASSERT_TRUE( r.angle.isSurfaceNormalA );
                ASSERT_FALSE( r.angle.isSurfaceNormalB );
            }
        }

        { // To half-infinite line.
            for ( bool backward : { false, true } )
            {
                auto b = toPrimitive( Line3f( Vector3f( 100.f + ( backward ? -1.f : 1.f ), 54, 10 ), Vector3f( 1, 0, 0 ) ) );
                b.negativeLength = 1;

                auto r = measure( a, b );

                ASSERT_EQ( r.angle.status, MeasureResult::Status::ok );

                ASSERT_EQ( r.angle.pointA, r.angle.pointB );
                Vector3f dir( std::sin( MR::PI_F / 3 ), 0.5f, 0 );
                Vector3f expectedPointA = a.center + Vector3f( dir * a.radius );
                ASSERT_LE( ( r.angle.pointA - expectedPointA ).length(), testEps );

                ASSERT_LE( ( r.angle.dirA - dir ).length(), testEps );
                ASSERT_LE( ( r.angle.dirB - b.dir ).length(), testEps );

                ASSERT_TRUE( r.angle.isSurfaceNormalA );
                ASSERT_FALSE( r.angle.isSurfaceNormalB );
            }
        }

        { // To finite line.
            for ( bool backward : { false, true } )
            {
                Vector3f lineStart( 100.f + ( backward ? -1.f : 1.f ), 54, 10 );
                auto b = toPrimitive( LineSegm3f( lineStart + Vector3f( 10, 0, 0 ), lineStart ) );

                auto r = measure( a, b );

                ASSERT_EQ( r.angle.status, MeasureResult::Status::ok );

                ASSERT_EQ( r.angle.pointA, r.angle.pointB );
                Vector3f dir( std::sin( MR::PI_F / 3 ), 0.5f, 0 );
                Vector3f expectedPointA = a.center + Vector3f( dir * a.radius );
                ASSERT_LE( ( r.angle.pointA - expectedPointA ).length(), testEps );

                ASSERT_LE( ( r.angle.dirA - dir ).length(), testEps );
                ASSERT_LE( ( r.angle.dirB - -b.dir ).length(), testEps ); // Inverted `dir` here.

                ASSERT_TRUE( r.angle.isSurfaceNormalA );
                ASSERT_FALSE( r.angle.isSurfaceNormalB );
            }
        }
    }

    { // No intersection.
        Primitives::Sphere a( Vector3f( 100, 50, 10 ), 8 );

        { // Line outside.
            auto b = toPrimitive( Line3f( Vector3f( 100, 58.01f, 10 ), Vector3f( 1, 0, 0 ) ) );

            auto r = measure( a, b );

            ASSERT_EQ( r.angle.status, MeasureResult::Status::badRelativeLocation );
        }

        { // Line segment inside.
            auto b = toPrimitive( LineSegm3f( Vector3f( 100, 52, 8 ), Vector3f( 100, 52, 12 ) ) );

            auto r = measure( a, b );

            ASSERT_EQ( r.angle.status, MeasureResult::Status::badRelativeLocation );
        }
    }
}

TEST( Features, Angle_Plane_Sphere )
{
    { // Reject a point.
        Primitives::Sphere point( Vector3f( 100, 50, 10 ), 0 );
        Primitives::Plane plane{ .center = Vector3f( 120, 58.01f, 10 ), .normal = Vector3f( 0, 1, 0 ) };

        auto r = measure( point, plane ).angle;
        ASSERT_EQ( r.status, MeasureResult::Status::badFeaturePair );
    }

    Primitives::Sphere sphere( Vector3f( 100, 50, 10 ), 8 );

    { // No collision.
        for ( bool sign : { false, true } )
        {
            Primitives::Plane plane{ .center = Vector3f( 120, 58.01f, 10 ), .normal = Vector3f( 0, sign ? -1.f : 1.f, 0 ) };
            ASSERT_EQ( measure( sphere, plane ).angle.status, MeasureResult::Status::badRelativeLocation );
        }
    }

    { // Collision.
        for ( bool sign : { false, true } )
        {
            Primitives::Plane plane{ .center = Vector3f( 120, 54, 10 ), .normal = Vector3f( 0, sign ? -1.f : 1.f, 0 ) };
            auto r = measure( sphere, plane ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_EQ( r.pointA, r.pointB );

            ASSERT_NEAR( r.pointA.y, 54, testEps );
            ASSERT_NEAR( ( r.pointA - Vector3f( 100, 54, 10 ) ).length(), std::sin( MR::PI_F / 3 ) * sphere.radius, testEps );

            ASSERT_LE( ( r.dirA - ( r.pointA - Vector3f( 100, 50, 10 ) ).normalized() ).length(), testEps );
            ASSERT_LE( ( r.dirB - Vector3f( 0, -1, 0 ) ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_TRUE( r.isSurfaceNormalB );
        }
    }

    { // Plane on sphere center.
        Primitives::Plane plane{ .center = Vector3f( 120, 50, 10 ), .normal = Vector3f( 0, 1, 0 ) };
        auto r = measure( sphere, plane ).angle;

        ASSERT_EQ( r.status, MeasureResult::Status::ok );

        ASSERT_EQ( r.pointA, r.pointB );

        ASSERT_NEAR( r.pointA.y, 50, testEps );
        ASSERT_NEAR( ( r.pointA - Vector3f( 100, 50, 10 ) ).length(), sphere.radius, testEps );

        ASSERT_LE( ( r.dirA - ( r.pointA - Vector3f( 100, 50, 10 ) ).normalized() ).length(), testEps );
        ASSERT_TRUE( r.dirB == Vector3f( 0, 1, 0 ) || r.dirB == Vector3f( 0, -1, 0 ) );

        ASSERT_TRUE( r.isSurfaceNormalA );
        ASSERT_TRUE( r.isSurfaceNormalB );
    }

    { // Exact center overlap.
        Primitives::Plane plane{ .center = Vector3f( 120, 50, 10 ), .normal = Vector3f( 0, 1, 0 ) };
        auto r = measure( sphere, plane ).angle;

        ASSERT_EQ( r.status, MeasureResult::Status::ok );

        ASSERT_EQ( r.pointA, r.pointB );

        ASSERT_NEAR( r.pointA.y, 50, testEps );
        ASSERT_NEAR( ( r.pointA - Vector3f( 100, 50, 10 ) ).length(), sphere.radius, testEps );

        ASSERT_LE( ( r.dirA - ( r.pointA - Vector3f( 100, 50, 10 ) ).normalized() ).length(), testEps );
        ASSERT_TRUE( r.dirB == Vector3f( 0, 1, 0 ) || r.dirB == Vector3f( 0, -1, 0 ) );

        ASSERT_TRUE( r.isSurfaceNormalA );
        ASSERT_TRUE( r.isSurfaceNormalB );
    }
}

TEST( Features, Angle_ConeSegment_ConeSegment )
{
    { // Line to line.
        auto a = toPrimitive( Line3f( Vector3f( 100, 50, 10 ), Vector3f( 1, 0, 0 ) ) );
        auto b = toPrimitive( Line3f( Vector3f( 101, 51, 20 ), Vector3f( 1, -1, 0 ) ) );

        auto r = measure( a, b ).angle;

        ASSERT_EQ( r.status, MeasureResult::Status::ok );

        ASSERT_LE( ( r.pointA - Vector3f( 102, 50, 10 ) ).length(), testEps );
        ASSERT_LE( ( r.pointB - Vector3f( 102, 50, 20 ) ).length(), testEps );

        // Here we accept the flipped direction as well. This test doesn't validate the direction sign selection logic.
        ASSERT_TRUE( ( r.dirA - a.dir ).length() < testEps || ( -r.dirA - a.dir ).length() < testEps );
        ASSERT_TRUE( ( r.dirB - b.dir ).length() < testEps || ( -r.dirB - b.dir ).length() < testEps );

        ASSERT_FALSE( r.isSurfaceNormalA );
        ASSERT_FALSE( r.isSurfaceNormalB );
    }
}

TEST( Features, Angle_Plane_ConeSegment )
{
    { // Line to plane.
        Primitives::Plane plane{ .center = Vector3f( 100, 50, 10 ), .normal = Vector3f( 1, 0, 0 ) };

        { // Intersecting.
            auto line = toPrimitive( LineSegm3f( Vector3f( 99, 60, 10 ), Vector3f( 102, 63, 10 ) ) );

            auto r = measure( plane, line ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_EQ( r.pointA, r.pointB );
            ASSERT_LE( ( r.pointA - Vector3f( 100, 61, 10 ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - line.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_FALSE( r.isSurfaceNormalB );
        }

        { // Non-intersecting, but we still extend to the intersection point.
            auto line = toPrimitive( LineSegm3f( Vector3f( 101, 60, 10 ), Vector3f( 104, 63, 10 ) ) );

            auto r = measure( plane, line ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_EQ( r.pointA, r.pointB );
            ASSERT_LE( ( r.pointA - Vector3f( 100, 59, 10 ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - line.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_FALSE( r.isSurfaceNormalB );
        }

        { // Parallel.
            auto line = toPrimitive( LineSegm3f( Vector3f( 101, 60, 10 ), Vector3f( 101, 63, 10 ) ) );

            auto r = measure( plane, line ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            Vector3f expectedPoint;
            ASSERT_TRUE( ( r.pointA - ( expectedPoint = Vector3f( 100, 60, 10 ) ) ).length() < testEps
                || ( r.pointA - ( expectedPoint = Vector3f( 100, 63, 10 ) ) ).length() < testEps
            );
            ASSERT_LE( ( r.pointB - ( expectedPoint + Vector3f( 1, 0, 0 ) ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - line.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_FALSE( r.isSurfaceNormalB );
        }
    }

    { // Circle to plane.
        Primitives::Plane plane{ .center = Vector3f( 100, 50, 10 ), .normal = Vector3f( 1, 0, 0 ) };

        { // Intersecting.
            auto circle = primitiveCircle( Vector3f( 101, 60, 10 ), Vector3f( 1, -1, 0 ), 4 );

            auto r = measure( plane, circle ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_EQ( r.pointA, r.pointB );
            ASSERT_LE( ( r.pointA - Vector3f( 100, 59, 10 ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - circle.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_TRUE( r.isSurfaceNormalB );
        }

        { // Non-intersecting, but we still extend to the intersection point.
            auto circle = primitiveCircle( Vector3f( 109, 60, 10 ), Vector3f( 4, -3, 0 ), 5 );

            auto r = measure( plane, circle ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_EQ( r.pointA, r.pointB );
            ASSERT_LE( ( r.pointA - Vector3f( 100, 48, 10 ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - circle.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_TRUE( r.isSurfaceNormalB );
        }

        { // Parallel.
            auto circle = primitiveCircle( Vector3f( 110, 60, 10 ), Vector3f( 1, 0, 0 ), 5 );

            auto r = measure( plane, circle ).angle;

            ASSERT_EQ( r.status, MeasureResult::Status::ok );

            ASSERT_NEAR( r.pointA.x, 100, testEps );
            ASSERT_NEAR( ( r.pointA - Vector3f( 100, 60, 10 ) ).length(), circle.positiveSideRadius, testEps );
            ASSERT_LE( ( r.pointB - ( r.pointA + Vector3f( 10, 0, 0 ) ) ).length(), testEps );

            ASSERT_LE( ( r.dirA - plane.normal ).length(), testEps );
            ASSERT_LE( ( r.dirB - circle.dir ).length(), testEps );

            ASSERT_TRUE( r.isSurfaceNormalA );
            ASSERT_TRUE( r.isSurfaceNormalB );
        }
    }
}

TEST( Features, Angle_Plane_Plane )
{
    Primitives::Plane a{ .center = Vector3f( 100, 50, 10 ), .normal = Vector3f( 1, 0, 0 ) };

    { // Intersecting.
        Primitives::Plane b{ .center = Vector3f( 102, 51, 10 ), .normal = Vector3f( 1, 1, 0 ).normalized() };

        auto r = measure( a, b ).angle;

        ASSERT_EQ( r.status, MeasureResult::Status::ok );

        ASSERT_EQ( r.pointA, r.pointB );
        ASSERT_LE( ( r.pointA - Vector3f( 100, 53, 10 ) ).length(), testEps );

        ASSERT_LE( ( r.dirA - Vector3f( 1, 0, 0 ) ).length(), testEps );
        ASSERT_LE( ( r.dirB - Vector3f( 1, 1, 0 ).normalized() ).length(), testEps );

        ASSERT_TRUE( r.isSurfaceNormalA );
        ASSERT_TRUE( r.isSurfaceNormalB );
    }
}

} // namespace MR::Features
