#include "MRFeatures.h"

#include "MRMesh/MRGTest.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"

// The tests in this file should be checked in debug builds too!

namespace MR::Features
{

Primitives::Sphere Primitives::ConeSegment::centerPoint() const
{
    bool posInf = std::isfinite( positiveLength );
    bool negInf = std::isfinite( negativeLength );
    assert( posInf == negInf );

    return toPrimitive( posInf || negInf ? center + dir * ( ( positiveLength - negativeLength ) / 2 ) : center );
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

    float shift = std::min( positiveSideRadius, negativeSideRadius ) * ( positiveLength + negativeLength ) / std::abs( positiveSideRadius - negativeSideRadius );
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
    return toPrimitive( center + dir * ( negative ? -negativeLength : positiveLength ) );
}

Primitives::Plane Primitives::ConeSegment::basePlane( bool negative ) const
{
    assert( std::isfinite( negative ? negativeLength : positiveLength ) );
    return Primitives::Plane( basePoint( negative ).center, negative ? -dir : dir );
}

Primitives::ConeSegment Primitives::ConeSegment::baseCircle( bool negative ) const
{
    assert( std::isfinite( negative ? negativeLength : positiveLength ) );

    ConeSegment ret = *this;
    ret.center = basePoint( negative ).center;
    ret.positiveLength = ret.negativeLength = 0;
    if ( negative )
        ret.dir = -ret.dir;
    return ret;
}

Primitives::ConeSegment primitiveCircle( const Vector3f& point, const Vector3f& normal, float rad )
{
    return {
        .center = point,
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
        .center = a,
        .dir = delta / ( deltaLen > 0 ? deltaLen : 1 ),
        .negativeSideRadius = rad,
        .positiveLength = deltaLen,
    };
}

std::optional<Primitives::Variant> primitiveFromObject( const Object& object )
{
    if ( auto point = dynamic_cast<const PointObject*>( &object ) )
    {
        return toPrimitive( point->parent()->worldXf()( point->getPoint() ) );
    }
    else if ( auto line = dynamic_cast<const LineObject*>( &object ) )
    {
        auto parentXf = line->parent()->worldXf();
        return toPrimitive( Line( parentXf( line->getCenter() ), parentXf.A * line->getDirection() ) );
    }
    else if ( auto plane = dynamic_cast<const PlaneObject*>( &object ) )
    {
        auto parentXf = plane->parent()->worldXf();
        return Primitives::Plane( parentXf( plane->getCenter() ), ( parentXf.A * plane->getNormal() ).normalized() );
    }
    else if ( auto sphere = dynamic_cast<const SphereObject*>( &object ) )
    {
        auto parentXf = sphere->parent()->worldXf();
        Vector3f scaleVec = parentXf.A.toScale();
        float scale = ( scaleVec.x + scaleVec.y + scaleVec.z ) / 3;
        return toPrimitive( Sphere( parentXf( sphere->getCenter() ), sphere->getRadius() * scale ) );
    }
    else if ( auto circle = dynamic_cast<const CircleObject*>( &object ) )
    {
        auto parentXf = circle->parent()->worldXf();
        Vector3f scaleVec = parentXf.A.toScale();
        float scale = ( scaleVec.x + scaleVec.y + scaleVec.z ) / 3;
        Primitives::ConeSegment ret{
            .center = parentXf( circle->getCenter() ),
            .dir = parentXf.A * circle->getNormal(),
            .positiveSideRadius = circle->getRadius() * scale,
            .negativeSideRadius = ret.positiveSideRadius,
            .hollow = true,
        };
        return ret;
    }
    else if ( auto cyl = dynamic_cast<const CylinderObject*>( &object ) )
    {
        auto parentXf = cyl->parent()->worldXf();
        Vector3f scaleVec = parentXf.A.toScale();
        float scale = ( scaleVec.x + scaleVec.y + scaleVec.z ) / 3;
        Primitives::ConeSegment ret{
            .center = parentXf( cyl->getCenter() ),
            .dir = parentXf.A * cyl->getDirection(),
            .positiveSideRadius = cyl->getRadius() * scale,
            .negativeSideRadius = ret.positiveSideRadius,
            .positiveLength = cyl->getLength() / 2 * scale,
            .negativeLength = ret.positiveLength,
            .hollow = true, // I guess?
        };
        return ret;
    }
    // TODO support cones.

    return {};
}

std::shared_ptr<VisualObject> primitiveToObject( const Primitives::Variant& primitive, float infiniteExtent )
{
    return std::visit( overloaded{
        []( const Primitives::Sphere& sphere ) -> std::shared_ptr<VisualObject>
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
        [infiniteExtent]( const Primitives::Plane& plane ) -> std::shared_ptr<VisualObject>
        {
            auto newPlane = std::make_shared<PlaneObject>();
            newPlane->setCenter( plane.center );
            newPlane->setNormal( plane.normal );
            newPlane->setSize( infiniteExtent );
            return newPlane;
        },
        [infiniteExtent]( const Primitives::ConeSegment& cone ) -> std::shared_ptr<VisualObject>
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
                    newLine->setCenter( cone.centerPoint().center );
                    newLine->setSize( ( posFinite ? cone.positiveLength + cone.negativeLength : infiniteExtent ) / 2 );
                }
                else
                {
                    newLine->setCenter( posFinite
                        ? cone.basePoint( false ).center - cone.dir * ( infiniteExtent / 2 )
                        : cone.basePoint( true ).center + cone.dir * ( infiniteExtent / 2 )
                    );
                    newLine->setSize( infiniteExtent / 2 );
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
                    newCylinder->setLength( posFinite ? cone.positiveLength + cone.negativeLength : infiniteExtent );
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

            // TODO support cones.
            return nullptr;
        },
    }, primitive );
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

DistanceResult Binary<Primitives::Sphere, Primitives::Sphere>::distance( const Primitives::Sphere& a, const Primitives::Sphere& b ) const
{
    DistanceResult ret;
    Vector3f dir = b.center - a.center;
    float dirLen = dir.length();
    ret.distance = dirLen - a.radius - b.radius;
    if ( dirLen > 0 )
        dir /= dirLen;
    else
        dir = Vector3f( 1, 0, 0 ); // An arbitrary default direction.
    ret.closestPointA = a.center + a.radius * dir;
    ret.closestPointB = b.center - b.radius * dir;
    return ret;
}

DistanceResult Binary<Primitives::ConeSegment, Primitives::Sphere>::distance( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const
{
    Vector3f centerDelta = b.center - a.center;

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
        axisToSphereCenterDir /= axisToSphereCenterDist;
    else
        axisToSphereCenterDir = cross( a.dir, a.dir.furthestBasisVector() ).normalized(); // An arbitrary direction.

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

    // Distance between the conical surface and the cone axis, measured along the normal from the conical surface to the spehre center.
    float axisToSurfaceSlopedDist = coneLengthIsFinite
        ? ( ( projectedSpherePos - negativeCapPos ) / ( positiveCapPos - negativeCapPos ) * ( a.positiveSideRadius - a.negativeSideRadius ) + a.negativeSideRadius ) * lengthFac
        : a.positiveSideRadius; // Either radius is fine there, they should be equal at this point.

    // Signed distance from the sphere center to the conical surface (positive if outside).
    float signedDistToSurface = axisToSphereCenterDist * lengthFac - axisToSurfaceSlopedDist;

    // Whether we're closer to the positive cap than the negative cap.
    bool positiveCapIsCloser = slopedSignedDistToPositiveCap >= slopedSignedDistToNegativeCap;

    // Signed distance from the sphere center to the closest cap (positive if outside).
    float signedDistToClosestCap = positiveCapIsCloser ? signedDistAlongAxis - a.positiveLength : -a.negativeLength - signedDistAlongAxis;

    DistanceResult ret;

    if ( a.hollow || signedDistToSurface > signedDistToClosestCap )
    {
        if ( signedDistToClosestCap <= 0 )
        {
            // Near the conical surface.
            ret.distance = ( a.hollow ? std::abs( signedDistToSurface ) : signedDistToSurface ) - b.radius;
            ret.closestPointA = a.center + a.dir * projectedSpherePos + normalToConicalSurface * axisToSurfaceSlopedDist;
            ret.closestPointB = b.center - normalToConicalSurface * b.radius * ( a.hollow && signedDistToSurface < 0 ? -1.f : 1.f );
            return ret;
        }
    }
    else
    {
        if ( signedDistToSurface <= 0 )
        {
            // Near the cap.
            ret.distance = signedDistToClosestCap - b.radius;
            ret.closestPointA = a.center + a.dir * ( positiveCapIsCloser ? a.positiveLength : -a.negativeLength ) + axisToSphereCenterDelta;
            ret.closestPointB = b.center - a.dir * ( ( positiveCapIsCloser ? 1 : -1 ) * b.radius );
            return ret;
        }
    }

    // Near the edge.

    // Distance from the sphere center to the cap edge, projected onto the normal to the cone axis.
    float distanceTowardsAxis = axisToSphereCenterDist - ( positiveCapIsCloser ? a.positiveSideRadius : a.negativeSideRadius );
    // Distance from the sphere center to the cap, projected onto the cone axis.
    float distanceAlongAxis = signedDistAlongAxis - ( positiveCapIsCloser ? a.positiveLength : -a.negativeLength );

    ret.distance = std::sqrt( distanceAlongAxis * distanceAlongAxis + distanceTowardsAxis * distanceTowardsAxis ) - b.radius;
    ret.closestPointA = a.center + a.dir * ( positiveCapIsCloser ? a.positiveLength : -a.negativeLength )
        + axisToSphereCenterDir * ( positiveCapIsCloser ? a.positiveSideRadius : a.negativeSideRadius );
    ret.closestPointB = b.center - ( a.dir * distanceAlongAxis + axisToSphereCenterDir * distanceTowardsAxis ).normalized() * b.radius;
    return ret;
}

DistanceResult Binary<Primitives::Plane, Primitives::Sphere>::distance( const Primitives::Plane& a, const Primitives::Sphere& b ) const
{
    float signedCenterDist = dot( a.normal, b.center - a.center );

    DistanceResult ret;
    ret.distance = std::abs( signedCenterDist ) - b.radius;
    ret.closestPointA = b.center - a.normal * signedCenterDist;
    ret.closestPointB = b.center - a.normal * ( b.radius * ( signedCenterDist >= 0 ? 1 : -1 ) );
    return ret;
}

DistanceResult Binary<Primitives::ConeSegment, Primitives::ConeSegment>::distance( const Primitives::ConeSegment& a, const Primitives::ConeSegment& b ) const
{
    if ( a.isZeroRadius() && b.isZeroRadius() )
    {
        // https://math.stackexchange.com/a/4764188

        Vector3f nDenorm = cross( a.dir, b.dir );
        Vector3f n = nDenorm.normalized();
        Vector3f centerDelta = b.center - a.center;

        float signedDist = dot( n, centerDelta );

        Vector3f bCenterFixed = b.center - n * signedDist;

        float tFac = 1.f / nDenorm.lengthSq();

        float ta = dot( cross( bCenterFixed - a.center, b.dir ), nDenorm ) * tFac;
        float tb = dot( cross( bCenterFixed - a.center, a.dir ), nDenorm ) * tFac;

        ta = std::clamp( ta, -a.negativeLength, a.positiveLength );
        tb = std::clamp( tb, -b.negativeLength, b.positiveLength );

        DistanceResult ret;
        ret.closestPointA = a.center + a.dir * ta;
        ret.closestPointB = b.center + b.dir * tb;
        ret.distance = ( ret.closestPointB - ret.closestPointA ).length();
        return ret;
    }
    else
    {
        // TODO: Support more cone types.
        return { .status = DistanceResult::Status::not_implemented };
    }
}

DistanceResult Binary<Primitives::Plane, Primitives::ConeSegment>::distance( const Primitives::Plane& a, const Primitives::ConeSegment& b ) const
{
    if ( !std::isfinite( b.positiveLength ) && !std::isfinite( b.negativeLength ) )
        return { .status = DistanceResult::Status::not_applicable };

    // A normal to the cone axis, parallel to the plane normal. The sign of this is unspecified.
    Vector3f sideDir = cross( cross( a.normal, b.dir ), b.dir ).normalized();
    if ( sideDir == Vector3f() || !sideDir.isFinite() )
        sideDir = cross( b.dir, b.dir.furthestBasisVector() ).normalized(); // An arbitrary direction.

    Vector3f positiveCapCenter = b.center + b.dir * b.positiveLength;
    Vector3f negativeCapCenter = b.center - b.dir * b.negativeLength;

    bool first = true;
    bool havePositivePoints = false, haveNegativePoints = false;

    float maxDist = 0, minDist = 0;
    Vector3f maxDistPoint, minDistPoint;

    for ( bool positiveSide : { true, false } )
    {
        if ( !std::isfinite( positiveSide ? b.positiveLength : b.negativeLength ) )
        {
            float dirDot = dot( a.normal, b.dir * ( positiveSide ? 1.f : -1.f ) );

            float dist = 0;
            if ( dirDot < 0.00001f ) // TODO move the epsilon to a constant?
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

    DistanceResult ret;
    if ( !havePositivePoints || ( haveNegativePoints && maxDist < -minDist ) )
    {
        ret.distance = maxDist;
        ret.closestPointB = maxDistPoint;
    }
    else
    {
        ret.distance = minDist;
        ret.closestPointB = minDistPoint;
    }

    ret.distance = std::abs( ret.distance );
    if ( havePositivePoints && haveNegativePoints )
        ret.distance = -ret.distance;

    ret.closestPointA = ret.closestPointB - a.normal * dot( a.normal, ret.closestPointB - a.center );

    return ret;
}

DistanceResult Binary<Primitives::Plane, Primitives::Plane>::distance( const Primitives::Plane& a, const Primitives::Plane& b ) const
{
    (void)a;
    (void)b;
    // We're not going to check for parallel-ness with some epsilon. You can just pick a point on one of the planes instead.
    return { .status = DistanceResult::Status::not_applicable };
}

} // namespace Traits

static constexpr float testEps = 0.0001f;

TEST( Features, PrimitiveConstruction )
{
    { // Infinite line to cone segment.
        Vector3f pos( 10, 20, 35 );
        Vector3f dir( 0, -3, 0 );
        auto cone = toPrimitive( Line3f( pos, dir ) );

        ASSERT_EQ( cone.positiveSideRadius, 0 );
        ASSERT_EQ( cone.negativeSideRadius, 0 );
        ASSERT_LE( ( cone.center - pos ).length(), testEps );
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
        ASSERT_LE( ( cone.center - pos ).length(), testEps );
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
        ASSERT_LE( ( cone.center - pos ).length(), testEps );
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
        ASSERT_LE( ( cone.center - pos ).length(), testEps );
        ASSERT_LE( ( cone.dir - Vector3f( 0, -1, 0 ) ).length(), testEps );
        ASSERT_NEAR( cone.positiveLength, 3, testEps );
        ASSERT_NEAR( cone.negativeLength, 0, testEps );
    }
}

TEST( Features, PrimitiveOps_ConeSegment )
{
    { // Untruncate.
        Primitives::ConeSegment cone{
            .center = Vector3f( 100, 50, 10 ),
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

TEST( Features, Sphere_Sphere )
{
    { // Point-point.
        { // Overlap.
            Vector3f a( 10, 20, 30 );
            auto r = distance( toPrimitive( a ), toPrimitive( a ) );
            ASSERT_NEAR( r.distance, 0, testEps );
            ASSERT_LE( ( r.closestPointA - a ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - a ).length(), testEps );
        }

        { // Positive distance.
            Vector3f a( 10, 20, 30 ), b( 7, 3, 1 );
            auto r = distance( toPrimitive( a ), toPrimitive( b ) );
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

            auto r = distance( sphere, sphere2 );
            ASSERT_NEAR( r.distance, -( sphere.radius + sphere2.radius ), testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + arbitraryDir * sphere.radius ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - arbitraryDir * sphere2.radius ) ).length(), testEps );
        }

        { // Negative distance.
            auto sphere2 = sphere;
            sphere2.radius = 4;
            float xOffset = 5;
            sphere2.center.x += xOffset;

            auto r = distance( sphere, sphere2 );
            ASSERT_NEAR( r.distance, xOffset - sphere.radius - sphere2.radius, testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + Vector3f( sphere.radius, 0, 0 ) ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - Vector3f( sphere2.radius, 0, 0 ) ) ).length(), testEps );
        }

        { // Positive distance.
            auto sphere2 = sphere;
            sphere2.radius = 4;
            float xOffset = 20;
            sphere2.center.x += xOffset;

            auto r = distance( sphere, sphere2 );
            ASSERT_NEAR( r.distance, xOffset - sphere.radius - sphere2.radius, testEps );
            ASSERT_LE( ( r.closestPointA - ( sphere.center + Vector3f( sphere.radius, 0, 0 ) ) ).length(), testEps );
            ASSERT_LE( ( r.closestPointB - ( sphere2.center - Vector3f( sphere2.radius, 0, 0 ) ) ).length(), testEps );
        }
    }
}

TEST( Features, ConeSegment_Sphere )
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
                line = Primitives::ConeSegment{ .center = linePos, .dir = lineDir, .positiveLength = lineStep * 2, .negativeLength = lineStep };
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
                        auto r = distance( line, sphere );
                        ASSERT_NEAR( r.distance, distToCap - sphereRad, testEps );
                        ASSERT_LE( ( r.closestPointA - closestPointOnLine ).length(), testEps );
                        ASSERT_NEAR( ( r.closestPointB - sphere.center ).length(), sphere.radius, testEps ); // Arbitrary direction here, due to an ambiguity.
                    }

                    { // Center doesn't overlap the line.
                        Vector3f offset( 1, 0, 3 );
                        Primitives::Sphere sphere( linePos + lineDir * lineStep * fac + offset, sphereRad );
                        auto r = distance( line, sphere );
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
                .center = Vector3f( 100, 200, 50 ),
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
                    pos += cone.center;
                    expectedPointOnCone += cone.center;
                    Primitives::Sphere sphere( pos, 3 );
                    auto r = distance( cone, sphere );
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
                        auto a = distance( cone, Primitives::Sphere( cone.center + cutoffPoint + capNormal * cutoffOffset, 1 ) );
                        ASSERT_LE( ( a.closestPointA - ( cone.center + Vector3f( point.x, 0, 0 ) ) ).length(), testEps );

                        // Closer to the surface.
                        auto b = distance( cone, Primitives::Sphere( cone.center + cutoffPoint - capNormal * cutoffOffset, 1 ) );
                        Vector3f expectedPoint = cone.center + point - dirToTip * point.y;
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
                            Primitives::Sphere sphere( cone.center + a * ( 1 - t ) + b * t, 1 );
                            auto r = distance( cone, sphere );
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

TEST( Features, Plane_Sphere )
{
    Vector3f planeCenter = Vector3f( 100, 50, 7 );
    Primitives::Plane plane( planeCenter, Vector3f( 1, 0, 0 ) );
    Vector3f sideOffset( 0, -13, 71 );

    for ( float dist : { -4.f, -2.f, 0.f, 2.f, 4.f } )
    {
        Primitives::Sphere sphere( planeCenter + sideOffset + plane.normal * dist, 3.f );
        auto r = distance( plane, sphere );

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

TEST( Features, ConeSegment_ConeSegment )
{
    { // Line-line.
        { // Infinite lines.
            { // Skew lines (not parallel and not intersecting).
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 10 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 ).normalized();

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = distance( l1, l2 );
                ASSERT_NEAR( r.distance, 10, testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 102, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - Vector3f( 102, 50, 20 ) ).length(), testEps );
            }

            { // An exact intersection.
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 0 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 ).normalized();

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = distance( l1, l2 );
                ASSERT_LE( r.distance, testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 102, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - r.closestPointA ).length(), testEps );
            }

            { // Parallel.
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 1, 0 ), d1( 1, 0, 0 ), d2 = d1;

                auto l1 = toPrimitive( Line3f( p1, d1 ) );
                auto l2 = toPrimitive( Line3f( p2, d2 ) );

                auto r = distance( l1, l2 );
                ASSERT_EQ( r.status, DistanceResult::Status::not_finite );
            }
        }

        { // Finite lines.
            { // Skew lines (not parallel and not intersecting).
                Vector3f p1( 100, 50, 10 ), p2 = p1 + Vector3f( 1, 2, 5 ), d1( 1, 0, 0 ), d2 = Vector3f( 1, -1, 0 );

                auto l1 = toPrimitive( LineSegm3f( p1, p1 + d1 ) );
                auto l2 = toPrimitive( LineSegm3f( p2 + d2, p2 ) ); // Backwards, why not.

                auto r = distance( l1, l2 );
                ASSERT_NEAR( r.distance, std::sqrt( 1 + 1 + 5*5 ), testEps );
                ASSERT_LE( ( r.closestPointA - Vector3f( 101, 50, 10 ) ).length(), testEps );
                ASSERT_LE( ( r.closestPointB - Vector3f( 102, 51, 15 ) ).length(), testEps );
            }
        }
    }
}

TEST( Features, Plane_ConeSegment )
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

                    auto r = distance( cone, plane );
                    ASSERT_NEAR( r.distance, expectedDist, testEps );

                    Vector3f slide;
                    ASSERT_TRUE(
                        ( r.closestPointA - surfacePoint ).length() < testEps ||
                        ( r.closestPointA - surfacePoint - ( slide = surfacePointSlideA ) ).length() < testEps ||
                        ( r.closestPointA - surfacePoint - ( slide = surfacePointSlideB ) ).length() < testEps
                    );
                    ASSERT_LE( ( r.closestPointB - closestPlanePoint - slide ).length(), testEps );
                };

                Vector3f randomPlaneCenterSlide = cross( offsetIntoCone, offsetIntoCone.furthestBasisVector() ) * 42.f;
                Primitives::Plane plane( closestPlanePoint + randomPlaneCenterSlide, offsetIntoCone );

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
            .center = Vector3f( 100, 50, 10 ),
            .dir = Vector3f( 1, 0, 0 ),
            .positiveSideRadius = 8,
            .negativeSideRadius = 14,
            .positiveLength = 20,
            .negativeLength = 10,
        };

        testPlane( cone, cone.center + Vector3f( 20, 0, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, 8, 0 ), Vector3f( 0, -8, 0 ) );

        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, -16, 0 ) );
        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( -1, -1, 0 ) );
        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( -1, -2, 0 ) );

        testPlane( cone, cone.center + Vector3f( -10, 14, 0 ), Vector3f( 1, -2, 0 ) );
        testPlane( cone, cone.center + Vector3f( -10, 14, 0 ), Vector3f( 1, -1, 0 ) );
        testPlane( cone, cone.center + Vector3f( -10, 14, 0 ), Vector3f( 1, 0, 0 ), Vector3f( 0, -28, 0 ) );

        testPlane( cone, cone.center + Vector3f( -10, 0, 0 ), Vector3f( 1, 0, 0 ), Vector3f( 0, 14, 0 ), Vector3f( 0, -14, 0 ) );
    }

    { // Half-infinite cone.
        Primitives::ConeSegment cone{
            .center = Vector3f( 100, 50, 10 ),
            .dir = Vector3f( 1, 0, 0 ),
            .positiveSideRadius = 8,
            .negativeSideRadius = 8,
            .positiveLength = 20,
            .negativeLength = INFINITY,
        };

        testPlane( cone, cone.center + Vector3f( 20, 0, 0 ), Vector3f( -1, 0, 0 ), Vector3f( 0, 8, 0 ), Vector3f( 0, -8, 0 ) );

        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( -1, -1, 0 ) );
        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( -100, -200, 0 ) );

        testPlane( cone, cone.center + Vector3f( 20, 8, 0 ), Vector3f( 0, -1, 0 ) );
        testPlane( cone, cone.center + Vector3f( -40, 8, 0 ), Vector3f( 0, -1, 0 ), Vector3f( 60, 0, 0 ) );
    }

    { // A line.
        { // Infinite.
            Primitives::ConeSegment line{
                .center = Vector3f( 100, 50, 10 ),
                .dir = Vector3f( 1, 0, 0 ),
                .positiveSideRadius = 0,
                .negativeSideRadius = 0,
                .positiveLength = INFINITY,
                .negativeLength = INFINITY,
            };

            Primitives::Plane plane( Vector3f( 1, 2, 3 ), Vector3f( 5, 6, 7 ) );

            auto r = distance( line, plane );
            ASSERT_EQ( r.status, DistanceResult::Status::not_applicable );
        }

        { // Finite.
            Primitives::ConeSegment line{
                .center = Vector3f( 100, 50, 10 ),
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
                .center = Vector3f( 100, 50, 10 ),
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

} // namespace MR::Features
