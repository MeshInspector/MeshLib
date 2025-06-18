#include <MRMesh/MRFeatures.h>
#include <MRMesh/MRLineSegm3.h>
#include <MRMesh/MRGTest.h>

namespace MR::Features
{

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
                            Primitives::Sphere sphere( cone.referencePoint + lerp( a, b, t ), 1 );
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

} //namespace MR::Features
