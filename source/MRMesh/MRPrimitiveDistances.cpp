#include "MRPrimitiveDistances.h"

#include "MRGTest.h"

namespace MR::PrimitiveDistances
{

Primitives::ConeSegment primitiveCircle( Vector3f point, Vector3f normal, float rad )
{
    return {
        .center = point,
        .dir = normal.normalized(),
        .positiveSideRadius = rad,
        .negativeSideRadius = rad,
    };
}

Primitives::ConeSegment primitiveCylinder( Vector3f a, Vector3f b, float rad )
{
    auto ret = primitiveCone( a, b, rad );
    ret.positiveSideRadius = ret.negativeSideRadius;
    return ret;
}

Primitives::ConeSegment primitiveCone( Vector3f a, Vector3f b, float rad )
{
    b -= a;
    float bLen = b.length();
    return {
        .center = a,
        .dir = b / ( bLen > 0 ? bLen : 1 ),
        .negativeSideRadius = rad,
        .positiveLength = bLen,
    };
}

namespace Traits
{

DistanceResult Distance<Primitives::Sphere, Primitives::Sphere>::operator()( const Primitives::Sphere& a, const Primitives::Sphere& b ) const
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

DistanceResult Distance<Primitives::ConeSegment, Primitives::Sphere>::operator()( const Primitives::ConeSegment& a, const Primitives::Sphere& b ) const
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

} // namespace Traits

static constexpr float testEps = 0.0001f;

TEST( PrimitiveDistances, PrimitiveConstruction )
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

TEST( PrimitiveDistances, Sphere_Sphere )
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

TEST( PrimitiveDistances, ConeSegment_Sphere )
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

} // namespace MR::PrimitiveDistances
