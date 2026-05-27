#include <MRMesh/MRDipole.h>
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRConstants.h>
#include "MRGTest.h"

namespace MR
{

/// see (6) in https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.pdf
static float triangleSolidAngle( const Vector3f & p, const Triangle3f & tri )
{
    Matrix3f m;
    m.x = tri[0] - p;
    m.y = tri[1] - p;
    m.z = tri[2] - p;
    auto x = m.x.length();
    auto y = m.y.length();
    auto z = m.z.length();
    auto den = x * y * z + dot( m.x, m.y ) * z + dot( m.y, m.z ) * x + dot( m.z, m.x ) * y;
    return 2 * std::atan2( m.det(), den );
}

TEST(MRMesh, TriangleSolidAngle)
{
    const Triangle3f tri =
    {
        Vector3f{ 0.0f, 0.0f, 0.0f },
        Vector3f{ 1.0f, 0.0f, 0.0f },
        Vector3f{ 0.0f, 1.0f, 0.0f }
    };
    const auto c = ( tri[0] + tri[1] + tri[2] ) / 3.0f;

    // solid angle near triangle center abruptly changes from -2pi to 2pi when the point crosses the triangle plane
    const auto x = triangleSolidAngle( c + Vector3f( 0, 0, 1e-5f ), tri );
    EXPECT_NEAR( x, -2 * PI_F, 1e-3f );
    auto y = triangleSolidAngle( c - Vector3f( 0, 0, 1e-5f ), tri );
    EXPECT_NEAR( y,  2 * PI_F, 1e-3f );

    // solid angle in triangle vertices is equal to zero exactly
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( triangleSolidAngle( tri[i], tri ), 0 );
    }

    // solid angle in the triangle plane outside of triangle is equal to zero exactly
    EXPECT_EQ( triangleSolidAngle( tri[1] + tri[2], tri ), 0 );
    EXPECT_EQ( triangleSolidAngle( -tri[1], tri ), 0 );
    EXPECT_EQ( triangleSolidAngle( -tri[2], tri ), 0 );
}

} //namespace MR
