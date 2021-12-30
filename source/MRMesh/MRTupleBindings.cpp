#include "MRTupleBindings.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, TupleBindings )
{    
    // Vector element type
    static_assert( std::is_same_v< std::tuple_element<0, Vector2f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector2f>::type, float> );

    static_assert( std::is_same_v< std::tuple_element<0, Vector2d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector2d>::type, double> );

    static_assert( std::is_same_v< std::tuple_element<0, Vector3f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector3f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<2, Vector3f>::type, float> );

    static_assert( std::is_same_v< std::tuple_element<0, Vector3d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector3d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<2, Vector3d>::type, double> );

    static_assert( std::is_same_v< std::tuple_element<0, Vector4f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector4f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<2, Vector4f>::type, float> );
    static_assert( std::is_same_v< std::tuple_element<3, Vector4f>::type, float> );

    static_assert( std::is_same_v< std::tuple_element<0, Vector4d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<1, Vector4d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<2, Vector4d>::type, double> );
    static_assert( std::is_same_v< std::tuple_element<3, Vector4d>::type, double> );

    // Matrix elements type
    static_assert( std::is_same_v< std::tuple_element<0, Matrix2f>::type, Vector2f> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix2f>::type, Vector2f> );

    static_assert( std::is_same_v< std::tuple_element<0, Matrix2d>::type, Vector2d> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix2d>::type, Vector2d> );

    static_assert( std::is_same_v< std::tuple_element<0, Matrix3f>::type, Vector3f> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix3f>::type, Vector3f> );
    static_assert( std::is_same_v< std::tuple_element<2, Matrix3f>::type, Vector3f> );

    static_assert( std::is_same_v< std::tuple_element<0, Matrix3d>::type, Vector3d> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix3d>::type, Vector3d> );
    static_assert( std::is_same_v< std::tuple_element<2, Matrix3d>::type, Vector3d> );

    static_assert( std::is_same_v< std::tuple_element<0, Matrix4f>::type, Vector4f> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix4f>::type, Vector4f> );
    static_assert( std::is_same_v< std::tuple_element<2, Matrix4f>::type, Vector4f> );
    static_assert( std::is_same_v< std::tuple_element<3, Matrix4f>::type, Vector4f> );

    static_assert( std::is_same_v< std::tuple_element<0, Matrix4d>::type, Vector4d> );
    static_assert( std::is_same_v< std::tuple_element<1, Matrix4d>::type, Vector4d> );
    static_assert( std::is_same_v< std::tuple_element<2, Matrix4d>::type, Vector4d> );
    static_assert( std::is_same_v< std::tuple_element<3, Matrix4d>::type, Vector4d> );

    // AffineXf elements type
    static_assert( std::is_same_v< std::tuple_element<0, AffineXf2f>::type, Matrix2f> );
    static_assert( std::is_same_v< std::tuple_element<1, AffineXf2f>::type, Vector2f> );

    static_assert( std::is_same_v< std::tuple_element<0, AffineXf2d>::type, Matrix2d> );
    static_assert( std::is_same_v< std::tuple_element<1, AffineXf2d>::type, Vector2d> );

    static_assert( std::is_same_v< std::tuple_element<0, AffineXf3f>::type, Matrix3f> );
    static_assert( std::is_same_v< std::tuple_element<1, AffineXf3f>::type, Vector3f> );

    static_assert( std::is_same_v< std::tuple_element<0, AffineXf3d>::type, Matrix3d> );
    static_assert( std::is_same_v< std::tuple_element<1, AffineXf3d>::type, Vector3d> );

    // Vector element size
    static_assert( std::tuple_size<Vector2f>::value == 2 );
    static_assert( std::tuple_size<Vector2d>::value == 2 );

    static_assert( std::tuple_size<Vector3f>::value == 3 );
    static_assert( std::tuple_size<Vector3d>::value == 3 );

    static_assert( std::tuple_size<Vector4f>::value == 4 );
    static_assert( std::tuple_size<Vector4d>::value == 4 );

    // Matrix element size
    static_assert( std::tuple_size<Matrix2f>::value == 2 );
    static_assert( std::tuple_size<Matrix2d>::value == 2 );

    static_assert( std::tuple_size<Matrix3f>::value == 3 );
    static_assert( std::tuple_size<Matrix3d>::value == 3 );

    static_assert( std::tuple_size<Matrix4f>::value == 4 );
    static_assert( std::tuple_size<Matrix4d>::value == 4 );

    // AffineXf element size
    static_assert( std::tuple_size<AffineXf2f>::value == 2 );
    static_assert( std::tuple_size<AffineXf2f>::value == 2 );

    static_assert( std::tuple_size<AffineXf2d>::value == 2 );
    static_assert( std::tuple_size<AffineXf2d>::value == 2 );

    // Id
    static_assert( std::tuple_size<FaceId>::value == 1 );
    static_assert( std::tuple_size<VertId>::value == 1 );
    static_assert( std::tuple_size<EdgeId>::value == 1 );

    static_assert( std::is_same_v< std::tuple_element<0, FaceId>::type, int> );
    static_assert( std::is_same_v< std::tuple_element<0, VertId>::type, int> );
    static_assert( std::is_same_v< std::tuple_element<0, EdgeId>::type, int> );

    // Color
    static_assert( std::tuple_size<Color>::value == 4 );

    static_assert( std::is_same_v< std::tuple_element<0, Color>::type, uint8_t> );
    static_assert( std::is_same_v< std::tuple_element<1, Color>::type, uint8_t> );
    static_assert( std::is_same_v< std::tuple_element<2, Color>::type, uint8_t> );
    static_assert( std::is_same_v< std::tuple_element<3, Color>::type, uint8_t> );

    // Get test
    Vector2f vec2f;
    auto& v2fx = get<0>( vec2f );
    const auto& v2fy = get<1>( vec2f );

    EXPECT_EQ( &v2fx, &vec2f.x );
    EXPECT_EQ( &v2fy, &vec2f.y );

    Vector3f vec3f;
    auto& v3fy = get<1>( vec3f );
    const auto& v3fz = get<2>( vec3f );

    EXPECT_EQ( &v3fy, &vec3f.y );
    EXPECT_EQ( &v3fz, &vec3f.z );

    Vector4f vec4f;
    auto& v4fz = get<2>( vec4f );
    const auto& v4fw = get<3>( vec4f );

    EXPECT_EQ( &v4fz, &vec4f.z );
    EXPECT_EQ( &v4fw, &vec4f.w );

    Matrix2d mat2d;
    auto& m2dx = get<0>( mat2d );
    const auto& m2dy = get<1>( mat2d );
    EXPECT_EQ( &m2dx, &mat2d.x );
    EXPECT_EQ( &m2dy, &mat2d.y );

    Matrix3d mat3d;
    auto& m3dy = get<1>( mat3d );
    const auto& m3dz = get<2>( mat3d );
    EXPECT_EQ( &m3dy, &mat3d.y );
    EXPECT_EQ( &m3dz, &mat3d.z );

    Matrix4d mat4d;
    auto& m4dz = get<2>( mat4d );
    const auto& m4dw = get<3>( mat4d );
    EXPECT_EQ( &m4dz, &mat4d.z );
    EXPECT_EQ( &m4dw, &mat4d.w );

    AffineXf2d a2d;
    auto& a2da = get<0>( a2d );
    const auto& a2db = get<1>( a2d );
    EXPECT_EQ( &a2da, &a2d.A );
    EXPECT_EQ( &a2db, &a2d.b );

    AffineXf2f a2f;
    const auto& a2fa = get<0>( a2f );
    auto& a2fb = get<1>( a2f );
    EXPECT_EQ( &a2fa, &a2f.A );
    EXPECT_EQ( &a2fb, &a2f.b );

    AffineXf3d a3d;
    auto& a3da = get<0>( a3d );
    const auto& a3db = get<1>( a3d );
    EXPECT_EQ( &a3da, &a3d.A );
    EXPECT_EQ( &a3db, &a3d.b );

    AffineXf3f a3f;
    const auto& a3fa = get<0>( a3f );
    auto& a3fb = get<1>( a3f );
    EXPECT_EQ( &a3fa, &a3f.A );
    EXPECT_EQ( &a3fb, &a3f.b );

    // MR::Id
    EXPECT_EQ( get<0>( 1_f ), 1 );
    EXPECT_EQ( get<0>( 2_v ), 2 );
    EXPECT_EQ( get<0>( 3_e ), 3 );

    auto fid = 1_f;
    EXPECT_EQ( get<0>( fid ), 1 );
    auto vid = 2_v;
    EXPECT_EQ( get<0>( vid ), 2 );
    auto eid = 3_e;
    EXPECT_EQ( get<0>( eid ), 3 );

    Color color;
    const auto& ccr = get<0>( color );
    const auto& ccg = get<1>( color );
    const auto& ccb = get<2>( color );
    const auto& cca = get<3>( color );
    auto& cr = get<0>( color );
    auto& cg = get<1>( color );
    auto& cb = get<2>( color );
    auto& ca = get<3>( color );
    EXPECT_EQ( &ccr, &color.r );
    EXPECT_EQ( &ccg, &color.g );
    EXPECT_EQ( &ccb, &color.b );
    EXPECT_EQ( &cca, &color.a );
    EXPECT_EQ( &cr, &color.r );
    EXPECT_EQ( &cg, &color.g );
    EXPECT_EQ( &cb, &color.b );
    EXPECT_EQ( &ca, &color.a );
}

}