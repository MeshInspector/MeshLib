#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRFormat.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMatrix4.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRVector4.h>

#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, Format )
{
    const Vector2f v2 { -1.f, +1.f };
    const auto v2Str = fmt::format( "{}", v2 );
    EXPECT_EQ( v2Str, "-1, 1" );
    const auto v2StrEx = fmt::format( "{:+.3f}", v2 );
    EXPECT_EQ( v2StrEx, "-1.000, +1.000" );

    const Vector3f v3 { -1.f, +1.f, -1.5f };
    const auto v3Str = fmt::format( "{}", v3 );
    EXPECT_EQ( v3Str, "-1, 1, -1.5" );
    const auto v3StrEx = fmt::format( "{:+.3f}", v3 );
    EXPECT_EQ( v3StrEx, "-1.000, +1.000, -1.500" );

    const Vector4f v4 { -1.f, +1.f, -1.5f, +1.5f };
    const auto v4Str = fmt::format( "{}", v4 );
    EXPECT_EQ( v4Str, "-1, 1, -1.5, 1.5" );
    const auto v4StrEx = fmt::format( "{:+.3f}", v4 );
    EXPECT_EQ( v4StrEx, "-1.000, +1.000, -1.500, +1.500" );

    const auto mat3 = Matrix3f::scale( -2.f );
    const auto mat3Str = fmt::format( "{}", mat3 );
    EXPECT_EQ( mat3Str, "{-2, 0, 0}, {0, -2, 0}, {0, 0, -2}" );

    const auto mat4 = Matrix4f::scale( +2.f );
    const auto mat4Str = fmt::format( "{:+}", mat4 );
    EXPECT_EQ( mat4Str, "{+2, +0, +0, +0}, {+0, +2, +0, +0}, {+0, +0, +2, +0}, {+0, +0, +0, +2}" );

    const AffineXf3f xf;
    const auto xfStr = fmt::format( "{}", xf );
    EXPECT_EQ( xfStr, "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 0, 0}" );

    const Box3f box3 { { -1.f, -1.f, 0.f }, { 0.f, +1.f, +1.f } };
    const auto box3Str = fmt::format( "{}", box3 );
    EXPECT_EQ( box3Str, "{-1, -1, 0}, {0, 1, 1}" );

    BitSet bs( 8 );
    bs.set( 4 );
    bs.set( 6 );
    const auto bsStr = fmt::format( "{}", bs );
    EXPECT_EQ( bsStr, "00001010" );
}

} // namespace MR
