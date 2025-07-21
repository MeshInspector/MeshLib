#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRSolidOfRevolution.h>

#include <gtest/gtest.h>

namespace
{

using namespace MR;

float sphereArea( float radius )
{
    return 4.f * PI_F * radius * radius;
}

float sphereVolume( float radius )
{
    return 4.f * PI_F * radius * radius * radius / 3.f;
}

float cylinderArea( float radius, float height )
{
    return 2.f * PI_F * radius * ( height + radius );
}

float cylinderVolume( float radius, float height )
{
    return PI_F * sqr( radius ) * height;
}

float coneArea( float radius, float height )
{
    return PI_F * radius * ( std::hypot( radius, height ) + radius );
}

float coneVolume( float radius, float height )
{
    return PI_F * sqr( radius ) * height / 3.f;
}

float torusArea( float majorRadius, float minorRadius )
{
    return 4.f * sqr( PI_F ) * majorRadius * minorRadius;
}

float torusVolume( float majorRadius, float minorRadius )
{
    return 2.f * sqr( PI_F ) * majorRadius * sqr( minorRadius );
}

} // namespace

namespace MR
{

TEST( MRMesh, makeSolidOfRevolution )
{
    const auto radius = 2.f;
    const auto height = 10.f;
    const auto resolution = 64;
    const auto errRatio = 0.005f;

    // sphere
    Contour2f sphereProfile;
    for ( auto i = 0; i < resolution; ++i )
    {
        const auto angle = PI_F * (float)i / (float)resolution;
        sphereProfile.emplace_back( radius * std::sin( angle ), -radius * std::cos( angle ) );
    }
    sphereProfile.emplace_back( 0.f, radius );

    const auto sphere = makeSolidOfRevolution( sphereProfile, resolution );
    EXPECT_EQ( sphere.topology.getValidVerts().count(), resolution * ( resolution - 1 ) + 2 );
    EXPECT_NEAR( sphere.area(), sphereArea( radius ), errRatio * sphereArea( radius ) );
    EXPECT_NEAR( sphere.volume(), sphereVolume( radius ), errRatio * sphereVolume( radius ) );

    const auto sphereBox = sphere.computeBoundingBox();
    for ( auto dim = 0; dim < sphereBox.elements; ++dim )
    {
        EXPECT_FLOAT_EQ( sphereBox.min[dim], -radius );
        EXPECT_FLOAT_EQ( sphereBox.max[dim], +radius );
    }

    // cylinder
    const Contour2f cylinderProfile {
        { 0.f, 0.f },
        { radius, 0.f },
        { radius, height },
        { 0.f, height },
    };

    const auto cylinder = makeSolidOfRevolution( cylinderProfile, resolution );
    EXPECT_EQ( cylinder.topology.getValidVerts().count(), 2 * resolution + 2 );
    EXPECT_NEAR( cylinder.area(), cylinderArea( radius, height ), errRatio * cylinderArea( radius, height ) );
    EXPECT_NEAR( cylinder.volume(), cylinderVolume( radius, height ), errRatio * cylinderVolume( radius, height ) );

    // cone
    const Contour2f coneProfile {
        { 0.f, 0.f },
        { radius, 0.f },
        { 0.f, height },
    };

    const auto cone = makeSolidOfRevolution( coneProfile, resolution );
    EXPECT_EQ( cone.topology.getValidVerts().count(), resolution + 2 );
    EXPECT_NEAR( cone.area(), coneArea( radius, height ), errRatio * coneArea( radius, height ) );
    EXPECT_NEAR( cone.volume(), coneVolume( radius, height ), errRatio * coneVolume( radius, height ) );

    // torus
    Contour2f torusProfile;
    for ( auto i = 0; i < resolution; ++i )
    {
        const auto angle = 2.f * PI_F * (float)i / (float)resolution;
        torusProfile.emplace_back( height + radius * std::cos( angle ), radius * std::sin( angle ) );
    }
    torusProfile.emplace_back( height + radius, 0.f );

    const auto torus = makeSolidOfRevolution( torusProfile, resolution );
    EXPECT_EQ( torus.topology.getValidVerts().count(), resolution * resolution );
    EXPECT_NEAR( torus.area(), torusArea( height, radius ), errRatio * torusArea( height, radius ) );
    EXPECT_NEAR( torus.volume(), torusVolume( height, radius ), errRatio * torusVolume( height, radius ) );
}

} // namespace MR
