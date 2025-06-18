#include "MRFeatures.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRLineSegm3.h"

namespace MR::Features
{

// Extracts a uniform scale factor from a matrix.
// If the scaling isn't actually uniform, returns some unspecified average scaling, which is hopefully better than just taking an arbitrary axis.
[[nodiscard]] static float getUniformScale( const Matrix3f& m )
{
    Matrix3f r, s;
    decomposeMatrix3( m, r, s );
    return ( s.x.x + s.y.y + s.z.z ) / 3;
}

Primitives::ConeSegment Primitives::Plane::intersectWithPlane( const Plane& other ) const
{
    Vector3f point = intersectWithLine( { .referencePoint = other.center, .dir = cross( other.normal, cross( other.normal, normal ) ).normalized() } ).center;
    return toPrimitive( Line( point, cross( normal, other.normal ) ) );
}

Primitives::ConeSegment toPrimitive( const Line3f& line )
{
    return { .referencePoint = line.p, .dir = line.d.normalized(), .positiveLength = INFINITY, .negativeLength = INFINITY };
}

Primitives::ConeSegment toPrimitive( const LineSegm3f& segm )
{
    return { .referencePoint = segm.a, .dir = segm.dir().normalized(), .positiveLength = segm.length() };
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
    if ( auto point = dynamic_cast<const PointObject*>( &object ) )
    {
        return toPrimitive( point->getPoint() );
    }
    else if ( auto line = dynamic_cast<const LineObject*>( &object ) )
    {
        return toPrimitive( LineSegm3f( line->getPointA(), line->getPointB() ) );
    }
    else if ( auto plane = dynamic_cast<const PlaneObject*>( &object ) )
    {
        return Primitives::Plane{ .center = plane->getCenter(), .normal = plane->getNormal()/* Already normalized. */ };
    }
    else if ( auto sphere = dynamic_cast<const SphereObject*>( &object ) )
    {
        return toPrimitive( Sphere( sphere->getCenter(), sphere->getRadius() ) );
    }
    else if ( auto circle = dynamic_cast<const CircleObject*>( &object ) )
    {
        float radius = circle->getRadius();
        return Primitives::ConeSegment{
            .referencePoint = circle->getCenter(),
            .dir = circle->getNormal(),
            .positiveSideRadius = radius,
            .negativeSideRadius = radius,
            .hollow = true,
        };
    }
    else if ( auto cyl = dynamic_cast<const CylinderObject*>( &object ) )
    {
        float radius = cyl->getRadius();
        float halfLen = cyl->getLength() / 2;
        return Primitives::ConeSegment{
            .referencePoint = cyl->getCenter(),
            .dir = cyl->getDirection(),
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
            .referencePoint = cone->getCenter(),
            .dir = -cone->getDirection(),
            .positiveSideRadius = 0,
            .negativeSideRadius = cone->getBaseRadius(),
            .positiveLength = 0,
            .negativeLength = cone->getHeight(),
            .hollow = true, // I guess?
        };
        return ret;
    }

    return {};
}

std::optional<Primitives::Variant> primitiveFromObjectWithWorldXf( const Object& object )
{
    auto ret = primitiveFromObject( object );
    if ( ret )
        *ret = transformPrimitive( object.worldXf(), *ret );
    return ret;
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

Primitives::Sphere transformPrimitive( const AffineXf3f& xf, const Primitives::Sphere& primitive )
{
    Primitives::Sphere ret;
    ret.center = xf( primitive.center );
    ret.radius = getUniformScale( xf.A ) * primitive.radius;
    return ret;
}

Primitives::Plane transformPrimitive( const AffineXf3f& xf, const Primitives::Plane& primitive )
{
    Primitives::Plane ret;
    ret.center = xf( primitive.center );
    ret.normal = ( xf.A.inverse().transposed() * primitive.normal ).normalized();
    return ret;
}

Primitives::ConeSegment transformPrimitive( const AffineXf3f& xf, const Primitives::ConeSegment& primitive )
{
    Primitives::ConeSegment ret;
    ret.referencePoint = xf( primitive.referencePoint );
    ret.dir = xf.A * primitive.dir;
    ret.hollow = primitive.hollow;

    float dirScale = ret.dir.length(); // Not dividing by `primitive.dir.length()`, that's supposed to be already normalized.
    ret.dir /= dirScale; // ...which ensures that this normalizes correctly without accumulating error.

    ret.positiveLength = primitive.positiveLength * dirScale;
    ret.negativeLength = primitive.negativeLength * dirScale;

    auto [n1, n2] = primitive.dir.perpendicular();
    n1 = xf.A * n1;
    n2 = xf.A * n2;
    float radiusScale = ( n1.length() + n2.length() ) / 2.f;

    ret.positiveSideRadius = primitive.positiveSideRadius * radiusScale;
    ret.negativeSideRadius = primitive.negativeSideRadius * radiusScale;

    return ret;
}

Primitives::Variant transformPrimitive( const AffineXf3f& xf, const Primitives::Variant& primitive )
{
    return std::visit( [&]( const auto& primitive ) -> Primitives::Variant { return (transformPrimitive)( xf, primitive ); }, primitive );
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
        return "N/A because of how the features are arranged";
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

} // namespace MR::Features
