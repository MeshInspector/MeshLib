#include <MRMesh/MRInt64Mul128.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>
#include <random>

namespace MR
{

TEST( MRMesh, Int64Mul128 )
{
    constexpr std::int64_t edge[] = { 0, 1, -1, 2, -2, INT64_MAX, INT64_MIN, INT64_MAX - 1, INT64_MIN + 1,
        std::int64_t( 1 ) << 62, -( std::int64_t( 1 ) << 62 ), 0x7FFFFFFF, -0x80000000LL };
    for ( auto a : edge )
        for ( auto b : edge )
            EXPECT_TRUE( Int64Mul128( a ) * Int64Mul128( b ) == FastInt128( a ) * FastInt128( b ) );

    std::mt19937_64 gen( 12345 );
    for ( int i = 0; i < 10000; ++i )
    {
        const auto a = std::int64_t( gen() );
        const auto b = std::int64_t( gen() );
        EXPECT_TRUE( Int64Mul128( a ) * Int64Mul128( b ) == FastInt128( a ) * FastInt128( b ) );

        // additions of Int64Mul128 stay 64-bit, so test them on values that cannot overflow
        const auto c = std::int64_t( std::int32_t( a ) );
        const auto d = std::int64_t( std::int32_t( b ) );
        EXPECT_EQ( Int64Mul128( c ) + Int64Mul128( d ), c + d );
        EXPECT_EQ( Int64Mul128( c ) - Int64Mul128( d ), c - d );
        EXPECT_EQ( -Int64Mul128( c ), -c );
    }
}

TEST( MRMesh, Int64Mul128Vector )
{
    std::mt19937_64 gen( 54321 );
    for ( int i = 0; i < 1000; ++i )
    {
        const Vector3i64 u{ std::int64_t( gen() ), std::int64_t( gen() ), std::int64_t( gen() ) };
        const Vector3i64 v{ std::int64_t( std::int32_t( gen() ) ), std::int64_t( std::int32_t( gen() ) ), std::int64_t( std::int32_t( gen() ) ) };

        EXPECT_TRUE( dot( Vector3i64mul{ u }, Vector3i64mul{ v } ) ==
            FastInt128( u.x ) * v.x + FastInt128( u.y ) * v.y + FastInt128( u.z ) * v.z );

        const Vector3i64 d = Vector3i64mul{ v } - Vector3i64mul{ v / 2 };
        EXPECT_EQ( d, v - v / 2 );
    }
}

} //namespace MR
