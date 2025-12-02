#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRFormat.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMatrix4.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRTriPoint.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRVector4.h>

#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, Format )
{
    Vector2f v2 { -1.f, +1.f };
    const auto v2Str = fmt::format( "{}", v2 );
    EXPECT_EQ( v2Str, "-1 1" );

    Vector3f v3 { -1.f, +1.f, 0.5f };
    const auto v3Str = fmt::format( "{}", v3 );
    EXPECT_EQ( v3Str, "-1 1 0.5" );

    Vector4f v4 { -1.f, +1.f, 0.5f, -0.5f };
    const auto v4Str = fmt::format( "{}", v4 );
    EXPECT_EQ( v4Str, "-1 1 0.5 -0.5" );

    const auto mat3 = Matrix3f::identity() * 3.f;
    const auto mat3Str = fmt::format( "{}", mat3 );
    EXPECT_EQ( mat3Str, "3 0 0\n0 3 0\n0 0 3\n" );
}

} // namespace MR
