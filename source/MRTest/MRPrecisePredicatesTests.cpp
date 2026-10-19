#include <MRMesh/MRPrecisePredicates2.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <MRMesh/MRPrecisePredicates3Poly.h>
#include <MRMesh/MRInSphere.h>
#include <MRMesh/MRSparsePolynomial.h>
#include <MRMesh/MRFastInt128.h>
#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRVarBigInt.h>
#include <MRMesh/MRBox.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <map>
#include <optional>
#include <random>
#include <vector>

namespace MR
{

TEST( MRMesh, doSegmentSegmentIntersect )
{
    std::array<PreciseVertCoords2, 4> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( -1,  0 ) }, //a
        PreciseVertCoords2{ 1_v, Vector2i{  1,  0 } }, //b

        PreciseVertCoords2{ 2_v, Vector2i(  0,  1 ) }, //c
        PreciseVertCoords2{ 3_v, Vector2i{  0, -1 } }  //d
    };

    auto res = doSegmentSegmentIntersect( vs );
    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( res.cIsLeftFromAB );

    std::swap( vs[2], vs[3] );
    res = doSegmentSegmentIntersect( vs );
    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( !res.cIsLeftFromAB );

    vs[3].pt.y = -5;
    res = doSegmentSegmentIntersect( vs );
    EXPECT_FALSE( res.doIntersect );
}

TEST( MRMesh, sosCCW )
{
    std::array<PreciseVertCoords2, 9> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{  0,  0 } },
        PreciseVertCoords2{ 1_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 2_v, Vector2i{  0,  1 } },
        PreciseVertCoords2{ 3_v, Vector2i{  0, -1 } },
        PreciseVertCoords2{ 4_v, Vector2i{  1,  0 } },
        PreciseVertCoords2{ 5_v, Vector2i{ -1,  0 } },
        PreciseVertCoords2{ 6_v, Vector2i{  0,  0 } },
        PreciseVertCoords2{ 7_v, Vector2i{  0,  1 } },
        PreciseVertCoords2{ 8_v, Vector2i{  0, -1 } }
    };

    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[2] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[3] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[4] } ) );
    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[5] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[6] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[2],vs[7] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[3],vs[8] } ) );
}

TEST( MRMesh, sosCCW2 )
{
    std::array<PreciseVertCoords2, 4> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ 1, 0 } },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 1 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ 0, 1 } },
        PreciseVertCoords2{ 3_v, Vector2i{ 1, 0 } }
    };

    EXPECT_FALSE( ccw( { vs[1],vs[0],vs[2] } ) );
    EXPECT_TRUE(  ccw( { vs[2],vs[3],vs[0] } ) );
}

TEST( MRMesh, sosSmaller2 )
{
    std::array<PreciseVertCoords2, 5> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ 0, 0 } },
        PreciseVertCoords2{ 1_v, Vector2i( 1, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ 0, 1 } },
        PreciseVertCoords2{ 3_v, Vector2i{ 0, 2 } },
        PreciseVertCoords2{ 4_v, Vector2i{ 0, 1 } } // vs[4].pt == vs[2].pt
    };

    // not-degenerate
    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[2], vs[3] } ) );
    EXPECT_FALSE( smaller2( { vs[0], vs[1], vs[3], vs[2] } ) );
    EXPECT_FALSE( smaller2( { vs[1], vs[0], vs[2], vs[3] } ) );

    // partially degenerate
    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[4], vs[2] } ) );
    EXPECT_FALSE( smaller2( { vs[0], vs[1], vs[2], vs[4] } ) );
    EXPECT_FALSE( smaller2( { vs[1], vs[0], vs[4], vs[2] } ) );
}

TEST( MRMesh, sosSmaller2FullDegen )
{
    std::array<PreciseVertCoords2, 4> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 0, 0 ) }
    };

    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[2], vs[3] } ) );

    // test that maximum degree in smaller2 can cope with most degenerate situation possible
    do
    {
        (void)smaller2( { vs[0], vs[1], vs[2], vs[3] } );
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, sosInCircle )
{
    std::array<PreciseVertCoords2, 4> vs =
    {
        PreciseVertCoords2{ 3_v, Vector2i{ -1, 2 } },
        PreciseVertCoords2{ 2_v, Vector2i( 0 , 0 ) },
        PreciseVertCoords2{ 0_v, Vector2i{ 3, 10 } },
        PreciseVertCoords2{ 1_v, Vector2i{ 0 , 0 } }
    };
    EXPECT_TRUE( ccw( { vs[0],vs[1],vs[2] } ) );

    // These 3 proves that vs[3] is inside vs[0]vs[1]vs[2] triangle
    EXPECT_TRUE( ccw( { vs[0],vs[1],vs[3] } ) );
    EXPECT_TRUE( ccw( { vs[1],vs[2],vs[3] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[0],vs[3] } ) );

    // Check that vs[3] is inCircle
    EXPECT_TRUE( inCircle( vs ) );
}

TEST( MRMesh, segmentIntersectionOrder2a )
{
    const std::array<PreciseVertCoords2, 6> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i(  0, -1 ) },
        PreciseVertCoords2{ 1_v, Vector2i(  0, -1 ) },

        PreciseVertCoords2{ 2_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 3_v, Vector2i(  0,  0 ) },

        PreciseVertCoords2{ 4_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 5_v, Vector2i(  1,  0 ) }
    };

    // both segments 03 and 12 intersect line segment 45
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[4], vs[5] } ).doIntersect );
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[1], vs[2], vs[4], vs[5] } ).doIntersect );

    // segments 03 and 12 intersect one with another
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[1], vs[2] } ).doIntersect );

    // intersection of 45 and 03 is closer to 4 than intersection of 45 and 12
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[3], vs[1], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[5], vs[4], vs[0], vs[3], vs[1], vs[2] } ) );

    // intersection of 45 and 03 is closer to 4 than intersection of 45 and 02
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[3], vs[0], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[2], vs[0], vs[3] } ) );
}

TEST( MRMesh, segmentIntersectionOrder2FullDegen )
{
    std::array<PreciseVertCoords2, 6> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 4_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 5_v, Vector2i( 0, 0 ) }
    };

    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ).doIntersect );
    EXPECT_TRUE(  doSegmentSegmentIntersect( { vs[0], vs[2], vs[1], vs[3] } ).doIntersect );
    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[1], vs[2] } ).doIntersect );

    // test that maximum degree in segmentIntersectionOrder can cope with most degenerate situation possible
    do
    {
        if ( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ).doIntersect
            && doSegmentSegmentIntersect( { vs[0], vs[1], vs[4], vs[5] } ).doIntersect )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, doSegmentSegmentIntersectPartialDegen )
{
    EXPECT_TRUE( doSegmentSegmentIntersect( {
        PreciseVertCoords2{ 0_v, { 0,  0} },
        PreciseVertCoords2{ 2_v, {-1, -1} },
        PreciseVertCoords2{ 1_v, {-1, -1} },
        PreciseVertCoords2{ 3_v, {-1, -1} } } ).doIntersect );

    EXPECT_TRUE( doSegmentSegmentIntersect( {
        PreciseVertCoords2{ 1_v, { 0,  0} },
        PreciseVertCoords2{ 2_v, {-2, -2} },
        PreciseVertCoords2{ 0_v, {-1, -1} },
        PreciseVertCoords2{ 3_v, {-1, -1} } } ).doIntersect );

    // degenerated segment with ends at last vertices (with smallest perturbation) never intersects anything
    for ( int x = -1; x <= 1; ++x )
        for ( int y = -1; y <= 1; ++y )
        {
            if ( x == 0 && y == 0 )
                continue;

            const PreciseVertCoords2 p2{ 2_v, {x, y} };
            const PreciseVertCoords2 p3{ 3_v, {x, y} };

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 0_v, {0, 0} },
                PreciseVertCoords2{ 1_v, {x, y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 1_v, {0, 0} },
                PreciseVertCoords2{ 0_v, {x, y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 0_v, {0, 0} },
                PreciseVertCoords2{ 1_v, {2*x, 2*y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 1_v, {0, 0} },
                PreciseVertCoords2{ 0_v, {2*x, 2*y} }, p2, p3 } ).doIntersect );
        }
}

TEST( MRMesh, sosInCircle2 )
{
    std::array<PreciseVertCoords2, 5> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ -106280744 , -1002263723 } },
        PreciseVertCoords2{ 1_v, Vector2i( -187288916 , -172107608 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ -25334363 , -1063004405 } },
        PreciseVertCoords2{ 3_v, Vector2i{ -15200618 , -10122159 } },
        PreciseVertCoords2{ 4_v, Vector2i{ -106280744 , -1002263723 } }
    };

    // Prove that 0_v 2_v 4_v circle is in +Y half plane (4_v 2_v is horde in lower part)
    EXPECT_FALSE( ccw( { vs[2],vs[4],vs[3] } ) ); // 3_v is to the right of 2-4 vec
    
    EXPECT_FALSE( inCircle( { vs[4],vs[2],vs[0],vs[3] } ) ); // 3_v is in circle

    // prove that 0_v is inside 142 triangle
    EXPECT_TRUE( ccw( { vs[1],vs[4],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[4],vs[2],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[1],vs[0] } ) );
    // it means that 142 circle should be larger in +Y half plane and so 3_v should be inside it
    EXPECT_FALSE( inCircle( { vs[1],vs[4],vs[2],vs[3] } ) );
}

TEST( MRMesh, inSphere )
{
    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;
    const auto On = InSphereResult::OnSphere;
    const auto NoS = InSphereResult::NoSphere;

    const Vector3i a{ 0, 0, 0 };
    const Vector3i b{ 2, 0, 0 };
    const Vector3i c{ 0, 2, 0 };
    // circumcircle of triangle ABC: center (1,1,0), squared radius 2, plane normal +Z

    // no sphere: the radius is less than the circumradius of ABC, collinear A, B, C
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, 1 }, 1 ), NoS );
    EXPECT_EQ( inSphere( a, Vector3i{ 1, 1, 1 }, Vector3i{ 2, 2, 2 }, Vector3i{ 0, 0, 1 }, 9 ), NoS );

    // no sphere: a triangle side is longer than the sphere's diameter
    EXPECT_EQ( inSphere( a, Vector3i{ 10, 0, 0 }, c, Vector3i{ 1, 1, 1 }, 4 ), NoS );
    EXPECT_EQ( inSphere( a, Vector3i{ 1, 0, 0 }, Vector3i{ -1, 0, 1 }, Vector3i{ 0, 0, 1 }, 1 ), NoS ); // the side BC

    // a side exactly equal to the diameter: the sphere can still exist (rSq equal to the squared circumradius)
    EXPECT_EQ( inSphere( a, b, Vector3i{ 1, 1, 0 }, Vector3i{ 1, 0, 0 }, 1 ), In );
    EXPECT_EQ( inSphere( a, b, Vector3i{ 1, 1, 0 }, Vector3i{ 1, 0, 1 }, 1 ), On );

    // rSq == 2: the unique sphere centered at (1,1,0)
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, 1 }, 2 ), In );
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 3, 3, 0 }, 2 ), Out );
    EXPECT_EQ( inSphere( a, b, c, b, 2 ), On ); // exactly on the sphere (rSq equal to the squared circumradius)

    // rSq == 4: sphere center at ( 1, 1, sqrt(2) )
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, 2 }, 4 ), In );
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, 3 }, 4 ), In );
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, -1 }, 4 ), Out );
    EXPECT_EQ( inSphere( a, b, c, Vector3i{ 1, 1, 4 }, 4 ), Out );

    // cyclic permutations of (A, B, C) keep the result, a swap selects the mirror sphere
    EXPECT_EQ( inSphere( b, c, a, Vector3i{ 1, 1, 2 }, 4 ), In );
    EXPECT_EQ( inSphere( c, a, b, Vector3i{ 1, 1, 2 }, 4 ), In );
    EXPECT_EQ( inSphere( b, a, c, Vector3i{ 1, 1, 2 }, 4 ), Out );
    EXPECT_EQ( inSphere( b, a, c, Vector3i{ 1, 1, -1 }, 4 ), In );

    // maximal magnitudes as if from getToIntConverter: no overflow in internal computations;
    // the difference of any two points must still fit in int, as the converter guarantees
    constexpr int H = 1'000'000'000;
    const Vector3i a1{ -H, -H, 0 }, b1{ H, -H, 0 }, c1{ -H, H, 0 };
    // circumcircle center (0,0,0), squared radius 2*H^2; with rSq = 3*H^2 sphere center is (0,0,H)
    const auto rSq = 3 * sqr( std::int64_t( H ) );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ 0, 0, 2 * H }, rSq ), In );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ 0, 0, -1 }, rSq ), In );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ H, H, 2 * H }, rSq ), On ); // exactly on the sphere
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ H, H, -H }, rSq ), Out ); // squared distance to the center 6*H^2
}

TEST( MRMesh, inSphereTester )
{
    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;
    const auto On = InSphereResult::OnSphere;

    const Vector3i a{ 0, 0, 0 };
    const Vector3i b{ 2, 0, 0 };
    const Vector3i c{ 0, 2, 0 };

    InSphereTesteri tester;
    EXPECT_FALSE( tester.reset( a, b, c, 1 ) ); // rSq below the squared circumradius
    EXPECT_FALSE( tester.reset( a, Vector3i{ 1, 1, 1 }, Vector3i{ 2, 2, 2 }, 9 ) ); // collinear
    EXPECT_FALSE( tester.reset( a, Vector3i{ 10, 0, 0 }, c, 4 ) ); // a side beyond the diameter

    ASSERT_TRUE( tester.reset( a, b, c, 4 ) ); // sphere center at ( 1, 1, sqrt(2) )
    EXPECT_EQ( tester( Vector3i{ 1, 1, 2 } ), In );
    EXPECT_EQ( tester( Vector3i{ 1, 1, -1 } ), Out );
    EXPECT_EQ( tester( Vector3i{ 30, 0, 0 } ), Out ); // farther than the diameter from A

    // same answers as one-call inSphere for a grid of query points
    for ( int z = -2; z <= 4; ++z )
        for ( int y = -2; y <= 4; ++y )
            for ( int x = -2; x <= 4; ++x )
            {
                const Vector3i d{ x, y, z };
                EXPECT_EQ( tester( d ), inSphere( a, b, c, d, 4 ) );
            }

    // reuse of the same tester for another sphere
    ASSERT_TRUE( tester.reset( a, b, c, 2 ) ); // the unique sphere centered at (1,1,0)
    EXPECT_EQ( tester( Vector3i{ 1, 1, 1 } ), In );
    EXPECT_EQ( tester( b ), On );
}

TEST( MRMesh, inSphereTesterFlip )
{
    const Vector3i a{ 0, 0, 0 }, b{ 2, 0, 0 }, c{ 0, 2, 0 };
    InSphereTesteri flipped, mirror;
    ASSERT_TRUE( flipped.reset( a, b, c, 4 ) );
    ASSERT_TRUE( mirror.reset( a, c, b, 4 ) );
    flipped.flip();

    // flip must give the answers of the tester reset with the swapped second and third points
    for ( int z = -2; z <= 4; ++z )
        for ( int y = -2; y <= 4; ++y )
            for ( int x = -2; x <= 4; ++x )
            {
                const Vector3i d{ x, y, z };
                EXPECT_EQ( flipped( d ), mirror( d ) );
            }
    flipped.flip();
    EXPECT_EQ( flipped( Vector3i{ 1, 1, 2 } ), InSphereResult::Inside ); // the initial sphere is back

    const Vector3d ad{ 0, 0, 0 }, bd{ 2, 0, 0 }, cd{ 0, 2, 0 };
    InSphereTesterd flippedd, mirrord;
    ASSERT_TRUE( flippedd.reset( ad, bd, cd, 4.0 ) );
    ASSERT_TRUE( mirrord.reset( ad, cd, bd, 4.0 ) );
    flippedd.flip();
    EXPECT_EQ( flippedd( Vector3d{ 1, 1, -1 } ), mirrord( Vector3d{ 1, 1, -1 } ) );

    // four concyclic points: every query point is exactly on the both spheres passing via three
    // others, so the ties of all the arrangements are resolved by ids and must survive the flip
    const PreciseVertCoords ps[4] = {
        { 0_v, Vector3i{  5, 0, 0 } },
        { 1_v, Vector3i{  0, 5, 0 } },
        { 2_v, Vector3i{ -5, 0, 0 } },
        { 3_v, Vector3i{  3, 4, 0 } }
    };
    InSphereTesterSoS flippedSos, mirrorSos;
    for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 4; ++j )
            for ( int k = 0; k < 4; ++k )
            {
                if ( i == j || j == k || i == k )
                    continue;
                ASSERT_TRUE( flippedSos.reset( ps[i], ps[j], ps[k], 169 ) );
                ASSERT_TRUE( mirrorSos.reset( ps[i], ps[k], ps[j], 169 ) );
                flippedSos.flip();
                const auto & d = ps[6 - i - j - k];
                EXPECT_EQ( flippedSos( d ), mirrorSos( d ) );
            }
}

TEST( MRMesh, fastInSphereTesterSoS )
{
    // the fast tester must answer exactly as InSphereTesterSoS does, only sooner
    InSphereTesterSoS exact;
    FastInSphereTesterSoS fast;

    // four concyclic points: every query is exactly on the sphere, so the fast path must always
    // defer to the exact one and the ties must still be resolved by the ids
    const PreciseVertCoords ps[4] = {
        { 0_v, Vector3i{  5, 0, 0 } },
        { 1_v, Vector3i{  0, 5, 0 } },
        { 2_v, Vector3i{ -5, 0, 0 } },
        { 3_v, Vector3i{  3, 4, 0 } }
    };
    for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 4; ++j )
            for ( int k = 0; k < 4; ++k )
            {
                if ( i == j || j == k || i == k )
                    continue;
                ASSERT_TRUE( exact.reset( ps[i], ps[j], ps[k], 169 ) );
                ASSERT_TRUE( fast.reset( ps[i], ps[j], ps[k], 169 ) );
                const auto & d = ps[6 - i - j - k];
                EXPECT_EQ( exact( d ), fast( d ) );
                exact.flip();
                fast.flip();
                EXPECT_EQ( exact( d ), fast( d ) );
            }

    // pseudo-random spheres and queries, from tiny coordinates up to the magnitudes of getToIntConverter
    std::uint64_t seed = 12345;
    auto rnd = [&seed]( int mag )
    {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return int( std::int64_t( seed >> 33 ) % ( 2 * mag + 1 ) ) - mag;
    };
    int tested = 0;
    for ( int mag : { 100, 1000000, 1000000000 } )
        for ( int t = 0; t < 300; ++t )
        {
            const PreciseVertCoords vs[3] = {
                { 0_v, Vector3i{ rnd( mag ), rnd( mag ), rnd( mag ) } },
                { 1_v, Vector3i{ rnd( mag ), rnd( mag ), rnd( mag ) } },
                { 2_v, Vector3i{ rnd( mag ), rnd( mag ), rnd( mag ) } }
            };
            const auto rSq = 4 * sqr( std::int64_t( mag ) );
            const bool ok = exact.reset( vs[0], vs[1], vs[2], rSq );
            ASSERT_EQ( ok, fast.reset( vs[0], vs[1], vs[2], rSq ) );
            if ( !ok )
                continue;
            // outsideBothSpheres must agree with the plain tester asked on both sides
            InSphereTesteri plain;
            ASSERT_TRUE( plain.reset( vs[0].pt, vs[1].pt, vs[2].pt, rSq ) );
            for ( int f = 0; f < 2; ++f )
            {
                for ( int q = 0; q < 10; ++q )
                {
                    const PreciseVertCoords d{ 3_v, Vector3i{ rnd( mag ), rnd( mag ), rnd( mag ) } };
                    EXPECT_EQ( exact( d ), fast( d ) );
                    const bool out0 = plain( d.pt ) == InSphereResult::Outside;
                    plain.flip();
                    const bool out1 = plain( d.pt ) == InSphereResult::Outside;
                    plain.flip();
                    EXPECT_EQ( fast.outsideBothSpheres( d.pt ), out0 && out1 );
                    ++tested;
                }
                exact.flip();
                fast.flip();
            }
        }
    EXPECT_GT( tested, 10000 );
}

TEST( MRMesh, inSphereTesterSoSSphereExists )
{
    // sphereExists must answer exactly as reset does, including the ties and the degenerate triangles,
    // which are frequent on the tiny grids with the radii of the same magnitude
    std::uint64_t seed = 12345;
    auto rnd = [&seed]( std::int64_t mag )
    {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return std::int64_t( seed >> 33 ) % ( 2 * mag + 1 ) - mag;
    };
    InSphereTesterSoS tester;
    int exists = 0;
    for ( int mag : { 1, 2, 3, 1000, 1000000000 } )
        for ( int t = 0; t < ( mag <= 3 ? 30 : 3000 ); ++t )
        {
            auto pt = [&] { return Vector3i{ int( rnd( mag ) ), int( rnd( mag ) ), int( rnd( mag ) ) }; };
            const PreciseVertCoords vs[3] = { { 0_v, pt() }, { 1_v, pt() }, { 2_v, pt() } };
            const auto rSq = std::abs( rnd( 3 * sqr( std::int64_t( mag ) ) ) );
            const bool e = tester.sphereExists( vs[0], vs[1], vs[2], rSq );
            EXPECT_EQ( e, tester.reset( vs[0], vs[1], vs[2], rSq ) );
            exists += e;
        }
    EXPECT_GT( exists, 4000 );
}

TEST( MRMesh, inSphereTesterFloat )
{
    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;

    const Vector3d a{ 0, 0, 0 }, b{ 2, 0, 0 }, c{ 0, 2, 0 };
    InSphereTesterd tester;
    EXPECT_FALSE( tester.reset( a, b, c, 1.0 ) ); // rSq below the squared circumradius
    ASSERT_TRUE( tester.reset( a, b, c, 4.0 ) ); // sphere center at ( 1, 1, sqrt(2) )
    EXPECT_EQ( tester( Vector3d{ 1, 1, 2 } ), In );
    EXPECT_EQ( tester( Vector3d{ 1, 1, -1 } ), Out );

    // same answers as one-call inSphere for a grid of query points
    for ( int z = -2; z <= 4; ++z )
        for ( int y = -2; y <= 4; ++y )
            for ( int x = -2; x <= 4; ++x )
            {
                const Vector3d d{ double( x ), double( y ), double( z ) };
                EXPECT_EQ( tester( d ), inSphere( a, b, c, d, 4.0 ) );
            }

    InSphereTesterf testerf;
    ASSERT_TRUE( testerf.reset( Vector3f{ 0, 0, 0 }, Vector3f{ 2, 0, 0 }, Vector3f{ 0, 2, 0 }, 4.0f ) );
    EXPECT_EQ( testerf( Vector3f{ 1, 1, 2 } ), In );
    EXPECT_EQ( testerf( Vector3f{ 1, 1, -1 } ), Out );
}

TEST( MRMesh, inSphereFloat )
{
    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;
    const auto On = InSphereResult::OnSphere;
    const auto NoS = InSphereResult::NoSphere;

    const Vector3d a{ 0, 0, 0 }, b{ 2, 0, 0 }, c{ 0, 2, 0 };
    // circumcircle of triangle (a,b,c): center (1,1,0), squared radius 2, plane normal +Z

    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 1, 1, 1 }, 1.0 ), NoS ); // radius below circumradius
    EXPECT_EQ( inSphere( a, Vector3d{ 1, 1, 1 }, Vector3d{ 2, 2, 2 }, Vector3d{ 0, 0, 1 }, 9.0 ), NoS ); // collinear

    // rSq == 2: the unique sphere centered at (1,1,0)
    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 1, 1, 1 }, 2.0 ), In );
    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 3, 3, 0 }, 2.0 ), Out );
    EXPECT_EQ( inSphere( a, b, c, b, 2.0 ), On ); // exactly on the sphere (small integers are exact in double)

    // rSq == 4: sphere center at ( 1, 1, sqrt(2) )
    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 1, 1, 2 }, 4.0 ), In );
    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 1, 1, -1 }, 4.0 ), Out );
    EXPECT_EQ( inSphere( a, b, c, Vector3d{ 1, 1, 4 }, 4.0 ), Out );
    EXPECT_EQ( inSphere( a, c, b, Vector3d{ 1, 1, -1 }, 4.0 ), In ); // a swap selects the mirror sphere

    // float instantiation
    EXPECT_EQ( inSphere( Vector3f{ 0, 0, 0 }, Vector3f{ 2, 0, 0 }, Vector3f{ 0, 2, 0 }, Vector3f{ 1, 1, 2 }, 4.0f ), In );
    EXPECT_EQ( inSphere( Vector3f{ 0, 0, 0 }, Vector3f{ 2, 0, 0 }, Vector3f{ 0, 2, 0 }, Vector3f{ 1, 1, -1 }, 4.0f ), Out );
}

TEST( MRMesh, inSphereTetrahedron )
{
    // four vertices of a tetrahedron inscribed in the cube [-1/2, 1/2]^3, sphere radius 1:
    // for every arrangement the query point must be outside the sphere if the triangle is oriented
    // outside the tetrahedron (its normal points away from the query), and inside otherwise
    const Vector3d ps[4] = {
        Vector3d{  0.5,  0.5, -0.5 },
        Vector3d{ -0.5,  0.5,  0.5 },
        Vector3d{  0.5,  0.5,  0.5 },
        Vector3d{  0.5, -0.5,  0.5 }
    };
    // the same points scaled x2 to integers, the radius scales to 2
    const Vector3i psi[4] = {
        Vector3i{  1,  1, -1 },
        Vector3i{ -1,  1,  1 },
        Vector3i{  1,  1,  1 },
        Vector3i{  1, -1,  1 }
    };
    int order[4] = { 0, 1, 2, 3 };
    int nInside = 0;
    do
    {
        const auto & a = ps[order[0]], & b = ps[order[1]], & c = ps[order[2]], & d = ps[order[3]];
        const bool expectInside = dot( d - a, cross( b - a, c - a ) ) > 0; // inside-oriented triangle
        const auto expected = expectInside ? InSphereResult::Inside : InSphereResult::Outside;
        EXPECT_EQ( inSphere( a, b, c, d, 1.0 ), expected );
        EXPECT_EQ( inSphere( psi[order[0]], psi[order[1]], psi[order[2]], psi[order[3]], 4 ), expected );
        nInside += expectInside ? 1 : 0;
    } while ( std::next_permutation( std::begin( order ), std::end( order ) ) );
    EXPECT_EQ( nInside, 12 ); // 4 outside-oriented triangles of 8 give Outside, each thrice (cyclic invariance)
}

TEST( MRMesh, sosInSphere )
{
    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;

    auto vc = []( VertId id, int x, int y, int z ) { return PreciseVertCoords{ id, Vector3i{ x, y, z } }; };

    // in all tie configurations below the query point vs[3] is exactly on the sphere;
    // the expected values are validated against exact rational-perturbation evaluation

    // sphere center (0,0,0), rSq = 25; the outcome depends on the assignment of ids
    EXPECT_EQ( inSphere( { vc( 0_v, 3,4,0 ), vc( 1_v, 4,0,3 ), vc( 2_v, 0,3,4 ), vc( 3_v, 0,0,5 ) }, 25 ), Out );
    EXPECT_EQ( inSphere( { vc( 3_v, 3,4,0 ), vc( 2_v, 4,0,3 ), vc( 1_v, 0,3,4 ), vc( 0_v, 0,0,5 ) }, 25 ), Out );
    EXPECT_EQ( inSphere( { vc( 1_v, 3,4,0 ), vc( 3_v, 4,0,3 ), vc( 2_v, 0,3,4 ), vc( 0_v, 0,0,5 ) }, 25 ), Out );
    EXPECT_EQ( inSphere( { vc( 2_v, 3,4,0 ), vc( 0_v, 4,0,3 ), vc( 3_v, 0,3,4 ), vc( 1_v, 0,0,5 ) }, 25 ), In );

    // the query point coincides with a triangle point (distinct ids)
    EXPECT_EQ( inSphere( { vc( 0_v, 3,4,0 ), vc( 1_v, 4,0,3 ), vc( 2_v, 0,3,4 ), vc( 3_v, 3,4,0 ) }, 25 ), In );
    EXPECT_EQ( inSphere( { vc( 3_v, 3,4,0 ), vc( 2_v, 4,0,3 ), vc( 1_v, 0,3,4 ), vc( 0_v, 3,4,0 ) }, 25 ), Out );

    // the first derivative vanishes for the smallest-id point: query coplanar with the center and two others
    EXPECT_EQ( inSphere( { vc( 0_v, -5,0,0 ), vc( 1_v, -4,-3,0 ), vc( 2_v, -4,0,-3 ), vc( 3_v, 4,0,3 ) }, 25 ), Out );
    EXPECT_EQ( inSphere( { vc( 0_v, -5,0,0 ), vc( 3_v, -4,-3,0 ), vc( 2_v, -4,0,-3 ), vc( 1_v, 4,0,3 ) }, 25 ), Out );

    // concyclic points on the tilted plane x+y+z=0, sphere center (1,1,1): sqrt( E*W ) is irrational
    EXPECT_EQ( inSphere( { vc( 0_v, 1,-2,1 ), vc( 1_v, 1,1,-2 ), vc( 2_v, -2,1,1 ), vc( 3_v, -1,2,-1 ) }, 9 ), Out );
    EXPECT_EQ( inSphere( { vc( 3_v, 1,-2,1 ), vc( 2_v, 1,1,-2 ), vc( 1_v, -2,1,1 ), vc( 0_v, -1,2,-1 ) }, 9 ), In );
    EXPECT_EQ( inSphere( { vc( 0_v, 1,-2,1 ), vc( 1_v, 1,1,-2 ), vc( 2_v, -2,1,1 ), vc( 3_v, 2,-1,-1 ) }, 9 ), In );
    EXPECT_EQ( inSphere( { vc( 0_v, -2,1,1 ), vc( 1_v, 2,-1,-1 ), vc( 2_v, 1,1,-2 ), vc( 3_v, -1,-1,2 ) }, 9 ), In );
    EXPECT_EQ( inSphere( { vc( 3_v, -2,1,1 ), vc( 2_v, 2,-1,-1 ), vc( 1_v, 1,1,-2 ), vc( 0_v, -1,-1,2 ) }, 9 ), Out );

    // rSq exactly equal to the squared circumradius in the plane z = 0: any perturbation
    // enlarges the circumradius, so the perturbed sphere does not exist
    EXPECT_EQ( inSphere( { vc( 0_v, 3,4,0 ), vc( 1_v, 5,0,0 ), vc( 2_v, 0,-5,0 ), vc( 3_v, -3,-4,0 ) }, 25 ), InSphereResult::NoSphere );

    // no tie: same answers as the plain overload (rSq = 4 > squared circumradius 2)
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 2,0,0 ), vc( 2_v, 0,2,0 ), vc( 3_v, 1,1,2 ) }, 4 ), In );
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 2,0,0 ), vc( 2_v, 0,2,0 ), vc( 3_v, 3,3,0 ) }, 4 ), Out );
    // while rSq exactly equal to the squared circumradius of this z = 0 triangle cannot survive
    // the perturbation even for the points strictly inside or outside
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 2,0,0 ), vc( 2_v, 0,2,0 ), vc( 3_v, 1,1,1 ) }, 2 ), InSphereResult::NoSphere );
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 2,0,0 ), vc( 2_v, 0,2,0 ), vc( 3_v, 3,3,0 ) }, 2 ), InSphereResult::NoSphere );

    // the same tie resolution via InSphereTesterSoS: the triangle ids are given once in reset
    InSphereTesterSoS tester;
    ASSERT_TRUE( tester.reset( vc( 0_v, 3,4,0 ), vc( 1_v, 4,0,3 ), vc( 2_v, 0,3,4 ), 25 ) );
    EXPECT_EQ( tester( vc( 3_v, 0,0,5 ) ), Out );
    ASSERT_TRUE( tester.reset( vc( 2_v, 3,4,0 ), vc( 0_v, 4,0,3 ), vc( 3_v, 0,3,4 ), 25 ) );
    EXPECT_EQ( tester( vc( 1_v, 0,0,5 ) ), In );
}

TEST( MRMesh, sosInSphereConcyclic )
{
    // four points on one circle lie on both spheres of radius sqrt(rSq) > circle radius passing via
    // any three of them, so every arrangement of the vertices is an exact tie resolved by ids only;
    // simulation-of-simplicity answers as for one perturbed configuration: over all 24 arrangements
    // of the same four vertices in the array, exactly half must give Inside
    auto countInside = []( std::array<PreciseVertCoords, 4> vs, std::int64_t rSq )
    {
        auto less = []( const PreciseVertCoords & l, const PreciseVertCoords & r ) { return l.id < r.id; };
        std::sort( vs.begin(), vs.end(), less );
        int cnt = 0;
        do
        {
            cnt += inSphere( vs, rSq ) == InSphereResult::Inside ? 1 : 0;
        }
        while ( std::next_permutation( vs.begin(), vs.end(), less ) );
        return cnt;
    };

    // circle x^2+y^2=25 in the plane z=0, sphere radius^2 = 169 (center at z=+-12)
    EXPECT_EQ( countInside( {
        PreciseVertCoords{ 0_v, Vector3i{  5, 0, 0 } },
        PreciseVertCoords{ 1_v, Vector3i{  0, 5, 0 } },
        PreciseVertCoords{ 2_v, Vector3i{ -5, 0, 0 } },
        PreciseVertCoords{ 3_v, Vector3i{  3, 4, 0 } } }, 169 ), 12 );

    // concyclic points on the tilted plane x+y+z=0 (circle radius^2 = 6), sqrt(E*W) is irrational
    EXPECT_EQ( countInside( {
        PreciseVertCoords{ 5_v, Vector3i{  1, -2,  1 } },
        PreciseVertCoords{ 2_v, Vector3i{ -2,  1,  1 } },
        PreciseVertCoords{ 8_v, Vector3i{  1,  1, -2 } },
        PreciseVertCoords{ 1_v, Vector3i{ -1,  2, -1 } } }, 9 ), 12 );
}

TEST( MRMesh, sosInSphereDegenerate )
{
    // full simulation-of-simplicity semantics in the degenerate configurations: the answers equal
    // the answers of the plain predicate for the symbolically perturbed points (see MRInSphere.h)

    const auto In = InSphereResult::Inside;
    const auto Out = InSphereResult::Outside;
    const auto NoS = InSphereResult::NoSphere;

    auto vc = []( VertId id, int x, int y, int z ) { return PreciseVertCoords{ id, Vector3i{ x, y, z } }; };

    // three (or four) coincident triangle points: the perturbed points form a needle-like triangle
    // with diverging circumradius, so the sphere never exists, independently of the ids
    EXPECT_EQ( inSphere( { vc( 0_v, 1,2,3 ), vc( 1_v, 1,2,3 ), vc( 2_v, 1,2,3 ), vc( 3_v, 1,2,3 ) }, 25 ), NoS );
    EXPECT_EQ( inSphere( { vc( 3_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 0_v, 0,0,0 ), vc( 2_v, 1,0,0 ) }, 25 ), NoS );

    // two coincident triangle points: the perturbed pair separates along z, so the sphere exists
    // iff 4*rSq*(Vx^2+Vy^2) > |V|^4 at the leading order, V = the third point minus the pair
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 4,0,0 ), vc( 3_v, 0,3,0 ) }, 25 ), Out ); // exists, the position is id-dependent:
    EXPECT_EQ( inSphere( { vc( 3_v, 0,0,0 ), vc( 2_v, 0,0,0 ), vc( 0_v, 4,0,0 ), vc( 1_v, 0,3,0 ) }, 25 ), In );
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 0,0,5 ), vc( 3_v, 0,3,0 ) }, 25 ), NoS ); // V along z: 4*rSq*(Vx^2+Vy^2) = 0 < |V|^4

    // rSq exactly equal to the squared circumradius: the perturbation decides the existence;
    // in the plane z = 0 any perturbation lifts a vertex and enlarges the circumradius
    EXPECT_EQ( inSphere( { vc( 0_v, 5,0,0 ), vc( 1_v, 0,5,0 ), vc( 2_v, -5,0,0 ), vc( 3_v, 0,0,0 ) }, 25 ), NoS ); // even for the center point
    EXPECT_EQ( inSphere( { vc( 0_v, 3,4,0 ), vc( 1_v, 5,0,0 ), vc( 2_v, 0,-5,0 ), vc( 3_v, -3,-4,0 ) }, 25 ), NoS );

    // on a tilted plane (x+y+z=0, circumradius^2 = 6 = rSq) the existence and the position
    // depend on the ids: all three outcomes on the same four points
    EXPECT_EQ( inSphere( { vc( 0_v, 1,-2,1 ), vc( 1_v, 1,1,-2 ), vc( 2_v, -2,1,1 ), vc( 3_v, -1,2,-1 ) }, 6 ), NoS );
    EXPECT_EQ( inSphere( { vc( 1_v, 1,-2,1 ), vc( 0_v, 1,1,-2 ), vc( 2_v, -2,1,1 ), vc( 3_v, -1,2,-1 ) }, 6 ), Out );
    EXPECT_EQ( inSphere( { vc( 2_v, 1,-2,1 ), vc( 1_v, 1,1,-2 ), vc( 3_v, -2,1,1 ), vc( 0_v, -1,2,-1 ) }, 6 ), In );

    // an exact tie of the pair existence rule: V = (3,4,5), 4*rSq*(Vx^2+Vy^2) == |V|^4 == 2500,
    // resolved by the deeper terms of the perturbation, so it depends on the ids
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 3,4,5 ), vc( 3_v, 3,4,0 ) }, 25 ), In );
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 3,4,5 ), vc( 3_v, 0,0,1 ) }, 25 ), Out );
    EXPECT_EQ( inSphere( { vc( 2_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 0_v, 3,4,5 ), vc( 3_v, 3,4,0 ) }, 25 ), NoS );

    // stable NoSphere answers: no perturbation creates the sphere
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 11,0,0 ), vc( 3_v, 0,3,0 ) }, 25 ), NoS ); // third point beyond the diameter
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 0,0,0 ), vc( 2_v, 10,0,0 ), vc( 3_v, 0,3,0 ) }, 25 ), NoS ); // third point exactly at the diameter
    EXPECT_EQ( inSphere( { vc( 0_v, 0,0,0 ), vc( 1_v, 1,0,0 ), vc( 2_v, 3,0,0 ), vc( 3_v, 0,1,0 ) }, 25 ), NoS );  // distinct collinear triangle

    // the tester resolves the existence at reset() already
    InSphereTesterSoS tester;
    EXPECT_FALSE( tester.reset( vc( 0_v, 5,0,0 ), vc( 1_v, 0,5,0 ), vc( 2_v, -5,0,0 ), 25 ) ); // planar rSq == squared circumradius
    EXPECT_FALSE( tester.reset( vc( 0_v, 1,2,3 ), vc( 1_v, 1,2,3 ), vc( 2_v, 1,2,3 ), 25 ) );  // coincident triple
    ASSERT_TRUE( tester.reset( vc( 3_v, 0,0,0 ), vc( 2_v, 0,0,0 ), vc( 0_v, 4,0,0 ), 25 ) );   // coincident pair, the sphere exists
    EXPECT_EQ( tester( vc( 1_v, 0,3,0 ) ), In );
    ASSERT_TRUE( tester.reset( vc( 2_v, 1,-2,1 ), vc( 1_v, 1,1,-2 ), vc( 3_v, -2,1,1 ), 6 ) ); // tilted rSq == squared circumradius
    EXPECT_EQ( tester( vc( 0_v, -1,2,-1 ) ), In );
}

TEST( MRMesh, segmentIntersectionOrder2Scale )
{
    // two segments crossing the third one at the same point (the origin): sa is the line x=0, sb is the line y=x;
    // the answer of a simulation-of-simplicity predicate must not depend on the scale of the coordinates
    // (and the coefficients of the polynomials at the large scale do not fit in 64 bits)
    const Vector2i pts[6] = { { -1, 0 }, { 1, 0 }, { 0,-1 }, { 0, 1 }, { -1,-1 }, { 1, 1 } };
    std::array<VertId, 6> ids;
    for ( int i = 0; i < 6; ++i )
        ids[i] = VertId( i );
    int mismatches = 0;
    do
    {
        int prev = -1;
        for ( int scale : { 1, 1 << 29 } )
        {
            std::array<PreciseVertCoords2, 6> vs;
            for ( int i = 0; i < 6; ++i )
                vs[i] = { ids[i], pts[i] * scale };
            if ( !doSegmentSegmentIntersect( { vs[2], vs[3], vs[0], vs[1] } ) || !doSegmentSegmentIntersect( { vs[4], vs[5], vs[0], vs[1] } ) )
                break;
            const int r = segmentIntersectionOrder( vs );
            mismatches += prev >= 0 && prev != r;
            prev = r;
        }
    }
    while ( std::next_permutation( ids.begin(), ids.end() ) );
    EXPECT_EQ( mismatches, 0 );
}

TEST( MRMesh, segmentIntersectionOrder2b )
{
    PreciseVertCoords2 vs[6] =
    {
        // s:
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 3, 0 ) },
        // sa:
        PreciseVertCoords2{ 2_v, Vector2i( 1,-1 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 1, 1 ) },
        // sb:
        PreciseVertCoords2{ 5_v, Vector2i( 2,-1 ) },
        PreciseVertCoords2{ 6_v, Vector2i( 2, 1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[5], vs[4] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[1], vs[0], vs[3], vs[2], vs[5], vs[4] } ) );

    // swapped sa and sb
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[3], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[4], vs[3], vs[2] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[1], vs[0], vs[5], vs[4], vs[3], vs[2] } ) );

    // shared point in sa and sb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[2], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[2], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[5], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[2], vs[5] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[5], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[4], vs[5], vs[2] } ) );
}

TEST( MRMesh, findTwoSegmentsIntersection )
{
    const Vector3i a( -100, -50 , 0 );
    const Vector3i b(  300, 150 , 0 );

    auto v = findTwoSegmentsIntersection( a, b, { 0, -200, 0 }, { 0, 100, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, -100, 0 }, { 0, 100, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, -100, 0 }, { 0, 300, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, 100, 0 }, { 0, 300, 0 } );
    EXPECT_FALSE( v.has_value() );

    // test with largest possible values
    constexpr int h = INT_MAX / 2 - 3;
    const Vector3i d( 1, 2, 3 );
    v = findTwoSegmentsIntersection( Vector3i{ h,  h, h } + d, Vector3i{ -h, -h, -h } + d,
                                     Vector3i{ h, -h, h } + d, Vector3i{ -h,  h, -h } + d );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, d );

    v = findTwoSegmentsIntersection( Vector3i{ h,  h, h } + d, Vector3i{ -h, -h, -h } + d,
                                     Vector3i{-h,  h,-h } + d, Vector3i{  h, -h,  h } + d );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, d );
}

TEST( MRMesh, findTriangleSegmentIntersectionPreciseInt )
{
    const Vector3i a( -100, -100, 0 );
    const Vector3i b(  200, -100, 0 );
    const Vector3i c( -100,  200, 0 );

    // the segment crosses the plane of the triangle at one third of its length from D
    auto v = findTriangleSegmentIntersectionPreciseInt( a, b, c, { 30, 60, -10 }, { 0, 0, 20 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i( 20, 40, 0 ) );

    // the segment lies in the plane of the triangle and crosses its side AB
    v = findTriangleSegmentIntersectionPreciseInt( a, b, c, { 0, -200, 0 }, { 0, 0, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i( 0, -100, 0 ) );

    // the segment lies in the plane of the triangle without crossing its sides
    v = findTriangleSegmentIntersectionPreciseInt( a, b, c, { 0, 0, 0 }, { 10, 10, 0 } );
    EXPECT_FALSE( v.has_value() );
}

TEST( MRMesh, orientParaboloid3d )
{
    // large numbers requiring more than 64-bit arithmetic, and degeneration (b==c)
    const Vector2i a{ 54209929, -710917541 };
    const Vector2i b{ 0, -365379885 };
    EXPECT_FALSE( orientParaboloid3d( a, b, b ) );
}

TEST( MRMesh, ccwAroundLine )
{
    const std::array<PreciseVertCoords, 5> vs =
    {
        PreciseVertCoords{ 0_v, Vector3i(  0,  0,  0 ) }, // the line is the z-axis directed up
        PreciseVertCoords{ 1_v, Vector3i(  0,  0,  1 ) },

        PreciseVertCoords{ 2_v, Vector3i(  2,  0,  7 ) }, // rotation 0 degrees
        PreciseVertCoords{ 3_v, Vector3i( -1,  2, -3 ) }, // rotation ~117 degrees
        PreciseVertCoords{ 4_v, Vector3i( -1, -2,  5 ) }  // rotation ~243 degrees
    };

    EXPECT_TRUE(  ccwAroundLine( vs ) );
    EXPECT_FALSE( ccwAroundLine( { vs[0], vs[1], vs[2], vs[4], vs[3] } ) ); // swapped half-planes
    EXPECT_FALSE( ccwAroundLine( { vs[1], vs[0], vs[2], vs[3], vs[4] } ) ); // reversed line
    EXPECT_TRUE(  ccwAroundLine( { vs[0], vs[1], vs[3], vs[4], vs[2] } ) ); // cyclic permutations
    EXPECT_TRUE(  ccwAroundLine( { vs[0], vs[1], vs[4], vs[2], vs[3] } ) );

    // all three half-planes within one half-space: rotations 0, ~11 and ~22 degrees
    const PreciseVertCoords a{ 2_v, Vector3i( 10, 0, 0 ) };
    const PreciseVertCoords b{ 3_v, Vector3i( 10, 2, 0 ) };
    const PreciseVertCoords c{ 4_v, Vector3i( 10, 4, 0 ) };
    EXPECT_TRUE(  ccwAroundLine( { vs[0], vs[1], a, b, c } ) );
    EXPECT_FALSE( ccwAroundLine( { vs[0], vs[1], a, c, b } ) );

    // the same three half-planes around an oblique line
    const PreciseVertCoords p{ 0_v, Vector3i( 1, 1, 1 ) };
    const PreciseVertCoords q{ 1_v, Vector3i( 3, 3, 3 ) };
    EXPECT_TRUE(  ccwAroundLine( { p, q, PreciseVertCoords{ 2_v, Vector3i(  1, -1,  0 ) },
                                         PreciseVertCoords{ 3_v, Vector3i(  0,  1, -1 ) },
                                         PreciseVertCoords{ 4_v, Vector3i( -1,  0,  1 ) } } ) );
    EXPECT_FALSE( ccwAroundLine( { p, q, PreciseVertCoords{ 2_v, Vector3i(  1, -1,  0 ) },
                                         PreciseVertCoords{ 4_v, Vector3i( -1,  0,  1 ) },
                                         PreciseVertCoords{ 3_v, Vector3i(  0,  1, -1 ) } } ) );
}

TEST( MRMesh, ccwAroundLineCircle )
{
    // 16 directions with strictly increasing rotation around the z-axis
    constexpr int cNum = 16;
    const Vector2i dirs[cNum] =
    {
        { 1, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 1 }, { -1, 2 }, { -1, 1 }, { -2, 1 },
        { -1, 0 }, { -2, -1 }, { -1, -1 }, { -1, -2 }, { 0, -1 }, { 1, -2 }, { 1, -1 }, { 2, -1 }
    };
    std::array<PreciseVertCoords, cNum> ps;
    for ( int i = 0; i < cNum; ++i )
        ps[i] = { VertId( i + 2 ), Vector3i( dirs[i].x, dirs[i].y, i - cNum / 2 ) }; // z varies on purpose

    const PreciseVertCoords p{ 0_v, Vector3i( 0, 0, -5 ) };
    const PreciseVertCoords q{ 1_v, Vector3i( 0, 0,  7 ) };
    for ( int i = 0; i < cNum; ++i )
        for ( int j = i + 1; j < cNum; ++j )
            for ( int k = j + 1; k < cNum; ++k )
            {
                EXPECT_TRUE(  ccwAroundLine( { p, q, ps[i], ps[j], ps[k] } ) );
                EXPECT_FALSE( ccwAroundLine( { p, q, ps[i], ps[k], ps[j] } ) );
            }
}

TEST( MRMesh, sosCcwAroundLine )
{
    std::array<PreciseVertCoords, 5> vs;
    for ( VertId i = 0_v; i < 5; ++i )
        vs[i].id = i; //and point coordinate is (0,0,0)

    // simulation-of-simplicity must answer consistently in the most degenerate situation possible
    do
    {
        const bool res = ccwAroundLine( vs );
        EXPECT_NE( res, ccwAroundLine( { vs[0], vs[1], vs[2], vs[4], vs[3] } ) ); // swapped half-planes
        EXPECT_NE( res, ccwAroundLine( { vs[1], vs[0], vs[2], vs[3], vs[4] } ) ); // reversed line
        EXPECT_EQ( res, ccwAroundLine( { vs[0], vs[1], vs[3], vs[4], vs[2] } ) ); // cyclic permutations
        EXPECT_EQ( res, ccwAroundLine( { vs[0], vs[1], vs[4], vs[2], vs[3] } ) );
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, doTriangleSegmentIntersect )
{
    const std::array<PreciseVertCoords, 5> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i(  2,  1, 0 ) }, //a
        PreciseVertCoords{ 1_v, Vector3i{ -2,  1, 0 } }, //b
        PreciseVertCoords{ 2_v, Vector3i{  0, -2, 0 } }, //c

        PreciseVertCoords{ 3_v, Vector3i{  0, 0, -1 } }, //d
        PreciseVertCoords{ 4_v, Vector3i{  0, 0,  1 } }  //e
    };

    auto res = doTriangleSegmentIntersect( vs );

    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( res.dIsLeftFromABC );
}

TEST( MRMesh, doTriangleSegmentIntersect2 )
{
    const std::array<PreciseVertCoords, 8> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i(  0, -1, -1 ) },
        PreciseVertCoords{ 1_v, Vector3i(  0, -1, -1 ) },

        PreciseVertCoords{ 2_v, Vector3i{  0, -1,  1 } },

        PreciseVertCoords{ 3_v, Vector3i{  0,  1,  1 } },

        PreciseVertCoords{ 4_v, Vector3i{  0, -1,  1 } },

        PreciseVertCoords{ 5_v, Vector3i{  0,  1,  1 } },

        PreciseVertCoords{ 6_v, Vector3i{  0,  0,  1 } },
        PreciseVertCoords{ 7_v, Vector3i{  1,  0,  1 } }
    };

    // both triangles 045 and 123 intersect line segment 67
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[6], vs[7] } ).doIntersect );
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[1], vs[2], vs[3], vs[6], vs[7] } ).doIntersect );

    // triangles 045 and 123 intersect one with another
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[1], vs[2] } ).doIntersect );
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[3], vs[1] } ).doIntersect );

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 123
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[1], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[7], vs[6], vs[0], vs[4], vs[5], vs[1], vs[2], vs[3] } ) );

    // one shared point 0: triangles 045 and 023 do not intersect one another
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[0], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[2], vs[3], vs[0], vs[4], vs[5] } ) );

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 143 (one shared point 4)
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[1], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[1], vs[4], vs[3], vs[0], vs[4], vs[5] } ) );

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 043 (two shared points 0 and 4)
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[0], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[3], vs[0], vs[4], vs[5] } ) );
}

TEST( MRMesh, doTriangleSegmentIntersectFullDegen )
{
    std::array<PreciseVertCoords, 5> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 2_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 3_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 4_v, Vector3i( 0, 0, 0 ) }
    };

    do
    {
        if ( vs[0].id < vs[1].id && vs[1].id < vs[2].id && vs[3].id < vs[4].id ) // ignore same triangles and segments with changed order of vertices
        {
            //spdlog::info( "{}{}{}x{}{}: {}", (int)vs[0].id, (int)vs[1].id, (int)vs[2].id, (int)vs[3].id, (int)vs[4].id,
            //    doTriangleSegmentIntersect( { vs[0], vs[1], vs[2], vs[3], vs[4] } ).doIntersect );
            EXPECT_EQ( doTriangleSegmentIntersect( { vs[0], vs[1], vs[2], vs[3], vs[4] } ).doIntersect, vs[3].id == 1 && vs[4].id == 3 );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, segmentIntersectionOrder3FullDegen )
{
    std::array<PreciseVertCoords, 8> vs;
    for ( VertId i = 0_v; i < 8; ++i )
        vs[i].id = i; //and point coordinate is (0,0,0)

    // test that maximum degree in segmentIntersectionOrder can cope with most degenerate situation possible

    // no shared vertices
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } )
         && doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );

    // one shared vertex
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } )
         && doTriangleSegmentIntersect( { vs[5], vs[6], vs[2], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[2] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end() - 1, []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, segmentIntersectionOrder3a )
{
    PreciseVertCoords vs[8] =
    {
        // s:
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 3, 0, 0 ) },
        // ta:
        PreciseVertCoords{ 2_v, Vector3i( 1,-1,-1 ) },
        PreciseVertCoords{ 3_v, Vector3i( 1, 1,-1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, 0, 1 ) },
        // tb:
        PreciseVertCoords{ 5_v, Vector3i( 2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 2, 0, 1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[6], vs[5], vs[7] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[1], vs[0], vs[3], vs[2], vs[4], vs[6], vs[5], vs[7] } ) );

    // swapped ta and tb
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[3], vs[4] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[7], vs[6], vs[2], vs[4], vs[3] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[1], vs[0], vs[5], vs[7], vs[6], vs[2], vs[4], vs[3] } ) );

    // one shared point in ta and tb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[4] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[6], vs[4], vs[5], vs[6], vs[7] } ) );

    // two shared points in ta and tb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[3], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[4], vs[3] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[6], vs[7], vs[5], vs[6], vs[7] } ) );
}

TEST( MRMesh, segmentIntersectionOrder3b )
{
    PreciseVertCoords vs[8] =
    {
        // s:
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 4, 0, 0 ) },
        // shared vertex of tris:
        PreciseVertCoords{ 2_v, Vector3i( 2, 0, 1 ) },
        // ta:
        PreciseVertCoords{ 3_v, Vector3i( 1,  100, -1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, -100, -1 ) },
        // tb:
        PreciseVertCoords{ 5_v, Vector3i( 3,    1, -1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2,   -1, -1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[2], vs[2], vs[3], vs[4] } ) );
}

TEST( MRMesh, segmentIntersectionOrder3c )
{
    // degenerate inputs reaching the polynomial path with 7 or 8 distinct vertex ids,
    // so the vertices with the largest ids must receive distinct (and huge) epsilon degrees
    auto check = []( const std::array<PreciseVertCoords, 8> & vs, bool expected )
    {
        ASSERT_TRUE( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) );
        ASSERT_TRUE( doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) );
        EXPECT_EQ( segmentIntersectionOrder( vs ), expected );
        // swapped triangles
        EXPECT_EQ( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[3], vs[4] } ), !expected );
        // reversed segment
        EXPECT_EQ( segmentIntersectionOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ), !expected );
    };

    // 8 distinct ids
    check( {
        PreciseVertCoords{ 12_v, Vector3i( 0,-1,-1 ) }, PreciseVertCoords{  1_v, Vector3i( 0, 1, 1 ) },
        PreciseVertCoords{ 11_v, Vector3i( 1, 0, 0 ) }, PreciseVertCoords{  5_v, Vector3i(-1,-1,-1 ) }, PreciseVertCoords{ 0_v, Vector3i( 1, 0, 0 ) },
        PreciseVertCoords{  4_v, Vector3i( 1, 0, 0 ) }, PreciseVertCoords{ 13_v, Vector3i(-1, 1, 1 ) }, PreciseVertCoords{ 6_v, Vector3i( 0,-1, 0 ) } }, true );

    // 7 distinct ids: one shared vertex in ta and tb
    check( {
        PreciseVertCoords{  8_v, Vector3i( 1,-1, 0 ) }, PreciseVertCoords{ 10_v, Vector3i( 1,-1, 1 ) },
        PreciseVertCoords{ 14_v, Vector3i( 1,-1, 1 ) }, PreciseVertCoords{  2_v, Vector3i( 0, 1, 0 ) }, PreciseVertCoords{ 4_v, Vector3i( 1, 1, 1 ) },
        PreciseVertCoords{  5_v, Vector3i( 1, 1, 1 ) }, PreciseVertCoords{  0_v, Vector3i( 0, 1, 1 ) }, PreciseVertCoords{ 14_v, Vector3i( 1,-1, 1 ) } }, false );
    check( {
        PreciseVertCoords{ 12_v, Vector3i( 0, 0, 0 ) }, PreciseVertCoords{  8_v, Vector3i( 0, 0,-1 ) },
        PreciseVertCoords{ 14_v, Vector3i( 0, 0, 0 ) }, PreciseVertCoords{  1_v, Vector3i( 0, 0,-1 ) }, PreciseVertCoords{ 0_v, Vector3i(-1, 1, 1 ) },
        PreciseVertCoords{  2_v, Vector3i( 1, 1,-1 ) }, PreciseVertCoords{  4_v, Vector3i(-1, 0, 1 ) }, PreciseVertCoords{ 14_v, Vector3i( 0, 0, 0 ) } }, false );
}

TEST( MRMesh, signOfProductsDiff )
{
    using P = SparsePolynomial<std::int64_t, int, 1000>;
    const P one( std::vector<P::Term>{ { 0, 1 } } );
    const P onePlusX( 1, 1, 1 );
    const P oneMinusX( 1, 1, -1 );
    const P a( 1, 1, 1, 2, 1 ); // 1 + x + x^2
    const P b( 1, 1, 2, 2, 3 ); // 1 + 2x + 3x^2 = the first three terms of a*a

    EXPECT_EQ( signOfProductsDiff<std::int64_t>( onePlusX, oneMinusX, one, one ), -1 ); // -x^2
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( one, one, onePlusX, oneMinusX ), 1 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( onePlusX, onePlusX, onePlusX, onePlusX ), 0 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( a, a, b, one ), 1 );  // 2x^3 + x^4
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( b, one, a, a ), -1 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( a, one, one, a ), 0 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( a, one, P{}, one ), 1 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( P{}, one, a, one ), -1 );
    EXPECT_EQ( signOfProductsDiff<std::int64_t>( P{}, P{}, P{}, P{} ), 0 );
}

TEST( MRMesh, lowestTermOfSumOfProducts )
{
    using P = SparsePolynomial<std::int64_t, int, 1000>;
    using S = SignedProduct<P, P>;
    const P one( std::vector<P::Term>{ { 0, 1 } } );
    const P a( 1, 1, 1, 2, 1 ); // 1 + x + x^2
    const P b( 1, 1, 2, 2, 3 ); // 1 + 2x + 3x^2 = the first three terms of a*a
    const P c( 0, 3, 2 ); // 2x^3, the next term of a*a
    using T = std::pair<int, int>;

    const S s1[3] = { { &a, &a, false }, { &b, &one, true }, { &c, &one, true } }; // x^4
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s1, 3 ), T( 4, 1 ) );
    EXPECT_EQ( lowestTermOfSumOfProducts<FastInt128>( s1, 3 ), T( 4, 1 ) ); // with the coefficients converted
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s1, 2 ), T( 3, 1 ) ); // 2x^3 + x^4
    const S s2[3] = { { &a, &a, false }, { &b, &one, true }, { &a, &c, true } }; // -x^4 - 2x^5
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s2, 3 ), T( 4, -1 ) );
    const S s3[2] = { { &a, &one, false }, { &one, &a, true } }; // zero
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s3, 2 ).second, 0 );
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s3, 0 ).second, 0 );

    // the products of two terms below degree `from` are skipped: they cancel one another here, so the answer stays
    const P d( 1, 1, 1, 2, 2 ); // 1 + x + 2x^2
    const S s4[2] = { { &a, &one, false }, { &one, &d, true } }; // -x^2
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s4, 2 ), T( 2, -1 ) );
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s4, 2, 2 ), T( 2, -1 ) );
    EXPECT_EQ( lowestTermOfProductsDiff<std::int64_t>( a, one, one, d, 2 ), T( 2, -1 ) );
    // they do not here, so only -c * one is left of s1
    EXPECT_EQ( lowestTermOfSumOfProducts<std::int64_t>( s1, 3, 3 ), T( 3, -1 ) );

    // factors with different coefficient types, multiplied in different types: the other branches of the multiplication
    using Q = SparsePolynomial<FastInt128, int, 1000>;
    const Q qOne( std::vector<Q::Term>{ { 0, FastInt128( 1 ) } } );
    const Q qa( FastInt128( 1 ), 1, FastInt128( 1 ), 2, FastInt128( 1 ) );
    const Q qb( FastInt128( 1 ), 1, FastInt128( 2 ), 2, FastInt128( 3 ) );
    const Q qc( FastInt128( 0 ), 3, FastInt128( 2 ) );
    const SignedProduct<P, Q> pq[3] = { { &a, &qa, false }, { &b, &qOne, true }, { &c, &qOne, true } }; // x^4 as s1
    EXPECT_EQ( lowestTermOfSumOfProducts<FastInt128>( pq, 3 ), T( 4, 1 ) ); // only the first factors converted
    const SignedProduct<Q, P> qp[3] = { { &qa, &a, false }, { &qb, &one, true }, { &qc, &one, true } };
    EXPECT_EQ( lowestTermOfSumOfProducts<FastInt128>( qp, 3 ), T( 4, 1 ) ); // only the second factors converted
    const SignedProduct<Q, Q> qq[3] = { { &qa, &qa, false }, { &qb, &qOne, true }, { &qc, &qOne, true } };
    EXPECT_EQ( ( lowestTermOfSumOfProducts<Int128Mul256, FastInt128>( qq, 3 ) ), T( 4, 1 ) ); // the factors multiplied in different types
    EXPECT_EQ( ( lowestTermOfSumOfProducts<FastInt128, Int128Mul256>( qq, 3 ) ), T( 4, 1 ) );

    // `from` takes no part in deducing the type of degrees, so an int works with int64 degrees
    using P64 = SparsePolynomial<std::int64_t, std::int64_t, 1000>;
    const P64 one64( std::vector<P64::Term>{ { 0, 1 } } );
    const P64 a64( 1, 1, 1, 2, 1 );
    const P64 d64( 1, 1, 1, 2, 2 );
    EXPECT_EQ( lowestTermOfProductsDiff<std::int64_t>( a64, one64, one64, d64, 2 ), ( std::pair<std::int64_t, int>( 2, -1 ) ) );
}

TEST( MRMesh, PerturbationStages )
{
    // ids 7, 3, 7, 5: the stages of 3, 5, 7 with the degrees 1, r(3), r(3) * r(5)
    const PreciseVertCoords vs[4] = { { 7_v, Vector3i( 1, 2, 3 ) }, { 3_v, Vector3i( 4, 5, 6 ) }, { 7_v, Vector3i( 1, 2, 3 ) }, { 5_v, Vector3i( 7, 8, 9 ) } };
    const std::int64_t radices[4] = { 10, 14, 10, 27 };
    const detail::PerturbationStages ps( vs, radices, 4 );
    ASSERT_EQ( ps.numStages(), 3 );
    EXPECT_EQ( ps.place( 1 ), 1 );
    EXPECT_EQ( ps.place( 2 ), 14 );
    EXPECT_EQ( ps.place( 3 ), 14 * 27 );
    EXPECT_EQ( ps[1].d, 1 );
    EXPECT_EQ( ps[3].d, 14 );
    EXPECT_EQ( ps[0].d, 14 * 27 );
    EXPECT_EQ( ps[2].d, 14 * 27 );
    EXPECT_EQ( ps[0].pt, Vector3i( 1, 2, 3 ) );
    EXPECT_EQ( ps.at( 0, 2 ).d, 0 ); // not perturbed yet
    EXPECT_EQ( ps.at( 0, 3 ).d, 14 * 27 );
    EXPECT_EQ( ps.at( 3, 3 ).d, 14 );
    EXPECT_TRUE( ps.startsAt( 3, 2 ) );
    EXPECT_FALSE( ps.startsAt( 3, 3 ) );

    // firstStage stops at the first value, and skips the stages with places above maxDeg
    int last = 0;
    EXPECT_EQ( ps.firstStage( [&] ( int s ) { last = s; return s == 2 ? std::optional<int>( 5 ) : std::nullopt; } ), 5 );
    EXPECT_EQ( last, 2 );
    EXPECT_FALSE( ps.firstStage( [&] ( int s ) { last = s; return std::optional<int>(); }, 100 ) );
    EXPECT_EQ( last, 2 );
    EXPECT_FALSE( ps.firstStage( [&] ( int s ) { last = s; return std::optional<int>(); } ) );
    EXPECT_EQ( last, 3 );
}

TEST( MRMesh, segmentIntersectionOrder3Scale )
{
    // two triangles crossing the segment at the same point (x=0): ta is the plane x=0, tb is the plane y=x;
    // the answer of a simulation-of-simplicity predicate must not depend on the scale of the coordinates
    // (and the coefficients of the polynomials at the large scale do not fit in 128 bits)
    const Vector3i pts[8] =
    {
        { -1, 0, 0 }, { 1, 0, 0 },             // s
        { 0,-1,-1 }, { 0, 1,-1 }, { 0, 0, 1 }, // ta
        { 1, 1,-1 }, {-1,-1,-1 }, { 0, 0, 1 }  // tb
    };
    std::array<VertId, 8> ids;
    for ( int i = 0; i < 8; ++i )
        ids[i] = VertId( i );
    int mismatches = 0, n = 0;
    do
    {
        if ( ++n % 101 != 0 ) // every permutation is too slow here, since all polynomials are full
            continue;
        int prev = -1;
        for ( int scale : { 1, 1 << 29 } )
        {
            std::array<PreciseVertCoords, 8> vs;
            for ( int i = 0; i < 8; ++i )
                vs[i] = { ids[i], pts[i] * scale };
            if ( !doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } )
              || !doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) )
                break;
            const int r = segmentIntersectionOrder( vs );
            mismatches += prev >= 0 && prev != r;
            prev = r;
        }
    }
    while ( std::next_permutation( ids.begin(), ids.end() ) );
    EXPECT_EQ( mismatches, 0 );
}

TEST( MRMesh, segmentIntersectionTriPlaneOrder )
{
    PreciseVertCoords vs[8] =
    {
        // s:
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 3, 0, 0 ) },
        // ta:
        PreciseVertCoords{ 2_v, Vector3i( 1,-1,-1 ) },
        PreciseVertCoords{ 3_v, Vector3i( 1, 1,-1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, 0, 1 ) },
        // tb:
        PreciseVertCoords{ 5_v, Vector3i( 2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 2, 0, 1 ) }
    };

    // pb is the plane of tb, s crosses it: the answer must be the same as for segmentIntersectionOrder
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[6], vs[5], vs[7] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[3], vs[4] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[5], vs[6], vs[7], vs[2], vs[4], vs[3] } ) );

    // one shared point in ta and pb
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[4] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], vs[5], vs[6], vs[4] } ) );

    // two shared points in ta and pb
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[3], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], vs[5], vs[4], vs[3] } ) );

    // plane pb crosses the line of s outside of s (beyond s[1] at x=5 or before s[0] at x=-2)
    PreciseVertCoords outer[6] =
    {
        PreciseVertCoords{ 5_v, Vector3i( 5,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 5, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 5, 0, 1 ) },
        PreciseVertCoords{ 5_v, Vector3i(-2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i(-2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i(-2, 0, 1 ) }
    };
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], outer[0], outer[1], outer[2] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], outer[1], outer[0], outer[2] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], outer[0], outer[1], outer[2] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], outer[3], outer[4], outer[5] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], outer[4], outer[3], outer[5] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], outer[3], outer[4], outer[5] } ) );

    // plane pb crosses triangle ta (general case): 3x+2z=4 meets the line of s at x=4/3 (after ta), 8x-6z=6 at x=3/4 (before ta)
    PreciseVertCoords tilted[6] =
    {
        PreciseVertCoords{ 5_v, Vector3i( 2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 0, 0, 2 ) },
        PreciseVertCoords{ 5_v, Vector3i( 0,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 0, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 3, 0, 3 ) }
    };
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], tilted[0], tilted[1], tilted[2] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], tilted[1], tilted[0], tilted[2] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], tilted[0], tilted[1], tilted[2] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], tilted[3], tilted[4], tilted[5] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], tilted[4], tilted[3], tilted[5] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], tilted[3], tilted[4], tilted[5] } ) );

    // s is parallel to pb (plane z=1 or plane y=1), and the perturbation of the points decides the order;
    // still reversing s must flip the answer, and the orientation of pb must not matter
    PreciseVertCoords par[6] =
    {
        PreciseVertCoords{ 5_v, Vector3i( 0,-1, 1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 0, 1, 1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 3, 0, 1 ) },
        PreciseVertCoords{ 5_v, Vector3i( 0, 1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 0, 1, 1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 3, 1, 0 ) }
    };
    for ( int i = 0; i < 6; i += 3 )
    {
        const bool r = segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], par[i], par[i+1], par[i+2] } );
        EXPECT_EQ( r,  segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], par[i+1], par[i], par[i+2] } ) );
        EXPECT_NE( r,  segmentIntersectionTriPlaneOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], par[i], par[i+1], par[i+2] } ) );
    }

    // 8 distinct ids with the three largest ones in pb: the polynomials here have the largest degrees that must be considered (cMaxPolyDTriPlane)
    const std::array<PreciseVertCoords, 8> deg =
    {
        PreciseVertCoords{  9_v, Vector3i( 1, 1,-1 ) }, PreciseVertCoords{  0_v, Vector3i( 0,-1, 1 ) },
        PreciseVertCoords{  4_v, Vector3i( 0, 1, 0 ) }, PreciseVertCoords{  1_v, Vector3i( 1,-1, 0 ) }, PreciseVertCoords{  6_v, Vector3i( 1, 1,-1 ) },
        PreciseVertCoords{ 14_v, Vector3i( 1,-1, 1 ) }, PreciseVertCoords{ 15_v, Vector3i( 1,-1, 1 ) }, PreciseVertCoords{ 12_v, Vector3i( 1,-1, 0 ) }
    };
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( deg ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { deg[1], deg[0], deg[2], deg[3], deg[4], deg[5], deg[6], deg[7] } ) );
}

TEST( MRMesh, segmentIntersectionTriPlaneOrderFullDegen )
{
    std::array<PreciseVertCoords, 8> vs;
    for ( VertId i = 0_v; i < 8; ++i )
        vs[i].id = i; //and point coordinate is (0,0,0)

    // test that maximum degree in segmentIntersectionTriPlaneOrder can cope with most degenerate situation possible

    // no shared vertices
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );

    // one shared vertex
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionTriPlaneOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[2] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end() - 1, []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

// some of the points at the origin and the others nearby, which the full degeneracy above does not cover: the terms of degrees
// up to the experimentally found limits do not decide some of these orders (one in 150 pairs of crossed triangles here with
// cMaxPolyDTriTri, as a counter in a temporary build found), and all terms do
TEST( MRMesh, segmentIntersectionOrder3PartialDegen )
{
    auto v = []( int id, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( 0, 0, z ) }; };
    const std::array<PreciseVertCoords, 8> vs = { v( 6, 0 ), v( 2, 0 ), v( 7, 0 ), v( 3, 0 ), v( 1, 0 ), v( 0, 0 ), v( 4, -1 ), v( 5, 0 ) };
    EXPECT_TRUE(  segmentIntersectionOrder( vs ) ); // not decided by the terms of degrees up to cMaxPolyDTriTri
    EXPECT_FALSE( segmentIntersectionOrder( { vs[1], vs[0], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[3], vs[4] } ) );
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( vs ) );

    auto u = []( int id, int x ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, 0, 0 ) }; };
    const std::array<PreciseVertCoords, 8> us = { u( 4, 1 ), u( 1, 0 ), u( 2, 0 ), u( 0, 1 ), u( 3, 1 ), u( 5, 1 ), u( 6, 1 ), u( 7, 1 ) };
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( us ) ); // not decided by the terms of degrees up to cMaxPolyDTriPlane
    EXPECT_TRUE(  segmentIntersectionTriPlaneOrder( { us[1], us[0], us[2], us[3], us[4], us[5], us[6], us[7] } ) );
    EXPECT_FALSE( segmentIntersectionTriPlaneOrder( { us[0], us[1], us[2], us[3], us[4], us[6], us[5], us[7] } ) );

    // random sets of the kind: both predicates give the same order where the segment crosses both triangles,
    // and the opposite one for the reversed segment and for the swapped triangles
    std::mt19937 gen( 1 );
    const Vector3i pts[] = { { 0, 0, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
#ifdef NDEBUG
    constexpr int cTrials = 200000;
#else
    constexpr int cTrials = 20000;
#endif
    int numOrders = 0, numPlanes = 0, numMismatches = 0;
    for ( int trial = 0; trial < cTrials; ++trial )
    {
        std::array<PreciseVertCoords, 8> ws;
        std::array<int, 8> ids = { 0, 1, 2, 3, 4, 5, 6, 7 };
        for ( int i = 7; i > 0; --i )
            std::swap( ids[i], ids[gen() % ( i + 1 )] );
        const int numPts = 1 + int( gen() % 6 );
        for ( int i = 0; i < 8; ++i )
            ws[i] = { VertId( ids[i] ), pts[gen() % numPts] };
        if ( gen() % 3 == 0 )
            ws[7] = ws[2]; // a vertex shared by the triangles
        if ( !doTriangleSegmentIntersect( { ws[2], ws[3], ws[4], ws[0], ws[1] } ) )
            continue;
        ++numPlanes;
        const bool p = segmentIntersectionTriPlaneOrder( ws );
        numMismatches += p == segmentIntersectionTriPlaneOrder( { ws[1], ws[0], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7] } );
        if ( !doTriangleSegmentIntersect( { ws[5], ws[6], ws[7], ws[0], ws[1] } ) )
            continue;
        ++numOrders;
        const bool o = segmentIntersectionOrder( ws );
        numMismatches += o != p;
        numMismatches += o == segmentIntersectionOrder( { ws[1], ws[0], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7] } );
        numMismatches += o == segmentIntersectionOrder( { ws[0], ws[1], ws[5], ws[6], ws[7], ws[2], ws[3], ws[4] } );
    }
    EXPECT_EQ( numMismatches, 0 );
    EXPECT_GT( numOrders, cTrials / 100 );
    EXPECT_GT( numPlanes, cTrials / 20 );
}

namespace
{

// a naive reference of simulation-of-simplicity for segmentIntersectionOrder and segmentIntersectionTriPlaneOrder, which shares
// no code with them: coordinate c of a point gets + t^( w_c * 27^r ) with w = 9, 3, 1 for x, y, z and r the place of the point's id
// among the different ids in ascending order (the perturbation that getPointDegrees defines), the volumes are expanded by cofactors
// as polynomials in t with big integer coefficients, and the sign of a polynomial for infinitesimal t > 0 is that of its lowest term
namespace SosReference
{

/// the not-zero coefficients of a polynomial in t by degree
using Poly = std::map<std::int64_t, VarBigInt>;

Poly add( const Poly & a, const Poly & b, int signB )
{
    Poly res = a;
    for ( const auto & [deg, c] : b )
        res[deg] = signB > 0 ? res[deg] + c : res[deg] - c;
    std::erase_if( res, []( const auto & t ) { return t.second.sign() == 0; } );
    return res;
}

Poly mul( const Poly & a, const Poly & b )
{
    Poly res;
    for ( const auto & [degA, cA] : a )
        for ( const auto & [degB, cB] : b )
            res[degA + degB] = res[degA + degB] + cA * cB;
    std::erase_if( res, []( const auto & t ) { return t.second.sign() == 0; } );
    return res;
}

/// the degree and the sign of the lowest term, sign 0 for the zero polynomial
std::pair<std::int64_t, int> lowest( const Poly & p )
{
    return p.empty() ? std::pair<std::int64_t, int>{ 0, 0 } : std::pair{ p.begin()->first, p.begin()->second.sign() };
}

struct Order
{
    bool taFirst = false;        ///< the answer of both predicates: whether the line of s=01 meets the plane of ta=234 before the plane of tb=567
    std::int64_t nomDegree = 0;  ///< the degree of the lowest term of the numerator of the difference of the two crossings along s
    std::int64_t diffDegree = 0; ///< the degree of the lowest term of the difference of the volumes of tb with s[0] and with s[1]
    bool crossesB = false;       ///< whether s crosses the plane of tb
};

Order order( const std::array<PreciseVertCoords, 8> & vs )
{
    std::vector<VertId> ids;
    for ( const auto & v : vs )
        ids.push_back( v.id );
    std::sort( ids.begin(), ids.end() );
    ids.erase( std::unique( ids.begin(), ids.end() ), ids.end() );

    std::array<std::array<Poly, 3>, 8> p; // the perturbed coordinates
    for ( int i = 0; i < 8; ++i )
    {
        std::int64_t place = 1;
        for ( auto it = ids.begin(); *it != vs[i].id; ++it )
            place *= 27;
        const std::int64_t w[3] = { 9, 3, 1 };
        for ( int c = 0; c < 3; ++c )
        {
            if ( vs[i].pt[c] != 0 )
                p[i][c][0] = VarBigInt( std::int64_t( vs[i].pt[c] ) );
            p[i][c][w[c] * place] = VarBigInt( std::int64_t( 1 ) );
        }
    }
    // the determinant of the rows a - d, b - d, c - d
    auto volume = [&] ( int a, int b, int c, int d )
    {
        std::array<Poly, 3> u, v, w;
        for ( int k = 0; k < 3; ++k )
        {
            u[k] = add( p[a][k], p[d][k], -1 );
            v[k] = add( p[b][k], p[d][k], -1 );
            w[k] = add( p[c][k], p[d][k], -1 );
        }
        const Poly minor0 = add( mul( v[1], w[2] ), mul( v[2], w[1] ), -1 );
        const Poly minor1 = add( mul( v[0], w[2] ), mul( v[2], w[0] ), -1 );
        const Poly minor2 = add( mul( v[0], w[1] ), mul( v[1], w[0] ), -1 );
        return add( add( mul( u[0], minor0 ), mul( u[1], minor1 ), -1 ), mul( u[2], minor2 ), 1 );
    };
    const Poly taOrg = volume( 2, 3, 4, 0 ), taDest = volume( 2, 3, 4, 1 );
    const Poly tbOrg = volume( 5, 6, 7, 0 ), tbDest = volume( 5, 6, 7, 1 );

    // the line of s meets the planes at the parameters taOrg / ( taOrg - taDest ) and tbOrg / ( tbOrg - tbDest ) from s[0] to s[1],
    // and the second minus the first is nom / ( ( taOrg - taDest ) * ( tbOrg - tbDest ) )
    const Poly nom = add( mul( taOrg, tbDest ), mul( tbOrg, taDest ), -1 );
    const auto [nomDegree, nomSign] = lowest( nom );
    const auto [diffDegree, diffSign] = lowest( add( tbOrg, tbDest, -1 ) );
    const int denSign = lowest( add( taOrg, taDest, -1 ) ).second * diffSign;
    EXPECT_NE( nomSign, 0 );
    EXPECT_NE( denSign, 0 );
    return
    {
        .taFirst = nomSign * denSign > 0,
        .nomDegree = nomDegree,
        .diffDegree = diffDegree,
        .crossesB = lowest( tbOrg ).second != lowest( tbDest ).second
    };
}

} // namespace SosReference

} // anonymous namespace

// both predicates agree with the naive reference above on random inputs: in general position, degenerate ones, and degenerate ones
// that the terms of degrees up to the limits (detail::cMaxPolyDTriTri, cMaxPolyDTriPlane) leave undecided
TEST( MRMesh, segmentIntersectionOrder3Reference )
{
    // the degree limits of the predicates
    constexpr std::int64_t cMaxDegTriTri = detail::cMaxPolyDTriTri, cMaxDegTriPlane = detail::cMaxPolyDTriPlane;
    std::mt19937 gen( 3 );
    auto shuffle = [&] ( std::array<int, 8> ids, int first, int last )
    {
        for ( int i = last - 1; i > first; --i )
            std::swap( ids[i], ids[first + gen() % ( i - first + 1 )] );
        return ids;
    };
    const Vector3i pool[] = { { 0, 0, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
#ifdef NDEBUG
    constexpr int cTrials = 4000;
#else
    constexpr int cTrials = 1000;
#endif
    int numPlanes = 0, numAboveTriPlane = 0, numOrders = 0, numExact = 0, numDegenerate = 0, numAboveTriTri = 0, numMismatches = 0;
    for ( int trial = 0; trial < cTrials; ++trial )
    {
        std::array<PreciseVertCoords, 8> ws;
        // the kinds of inputs below in the ratio 1 : 2 : 3 : 2 (the volumes of points in general position take longest to expand)
        constexpr int cKinds[8] = { 0, 1, 1, 2, 2, 2, 3, 3 };
        switch ( cKinds[trial % 8] )
        {
        case 0:
        {
            // points in general position: a segment along the z-axis and triangles around it
            const auto ids = shuffle( { 0, 1, 2, 3, 4, 5, 6, 7 }, 0, 8 );
            for ( int i = 0; i < 8; ++i )
            {
                const int r = i < 2 ? 1 : 3;
                const int x = int( gen() % ( 2 * r + 1 ) ) - r;
                const int y = int( gen() % ( 2 * r + 1 ) ) - r;
                const int z = i < 2 ? 3 - 6 * i : int( gen() % 5 ) - 2;
                ws[i] = { VertId( ids[i] ), Vector3i( x, y, z ) };
            }
            break;
        }
        case 1:
        {
            // the points of a small pool, sometimes with a vertex shared by the triangles
            const auto ids = shuffle( { 0, 1, 2, 3, 4, 5, 6, 7 }, 0, 8 );
            const int numPts = 1 + int( gen() % 6 );
            for ( int i = 0; i < 8; ++i )
                ws[i] = { VertId( ids[i] ), pool[gen() % numPts] };
            if ( gen() % 3 == 0 )
                ws[7] = ws[2];
            break;
        }
        case 2:
        {
            // all points on the z-axis, the segment with ids 2 and 7, and the triangles with ids 3, 6 and with 4, 5, each with one of 0, 1:
            // segmentIntersectionOrder of a quarter of them needs the terms above cMaxDegTriTri
            std::array<int, 8> ids = { 2, 7, 0, 3, 6, 1, 4, 5 };
            if ( gen() % 2 )
                std::swap( ids[0], ids[1] );
            if ( gen() % 2 )
                std::swap( ids[2], ids[5] );
            ids = shuffle( shuffle( ids, 2, 5 ), 5, 8 );
            if ( gen() % 2 )
                std::swap_ranges( ids.begin() + 2, ids.begin() + 5, ids.begin() + 5 );
            for ( int i = 0; i < 8; ++i )
                ws[i] = { VertId( ids[i] ), Vector3i( 0, 0, int( gen() % 2 ) ) };
            break;
        }
        default:
        {
            // all points on the x-axis: the segment from 0 to 1 with two of the ids 0 to 4, the smaller one at 0, triangle ta with
            // the other three, and the plane of tb through three points at 1 with the largest ids:
            // segmentIntersectionTriPlaneOrder of two fifths of them needs the terms above cMaxDegTriPlane
            const auto ids = shuffle( shuffle( { 0, 1, 2, 3, 4, 5, 6, 7 }, 0, 5 ), 5, 8 );
            for ( int i = 0; i < 8; ++i )
                ws[i] = { VertId( ids[i] ), Vector3i( i < 5 ? int( gen() % 2 ) : 1, 0, 0 ) };
            ws[0].pt.x = ids[0] < ids[1] ? 0 : 1;
            ws[1].pt.x = 1 - ws[0].pt.x;
            break;
        }
        }
        if ( !doTriangleSegmentIntersect( { ws[2], ws[3], ws[4], ws[0], ws[1] } ) )
            continue;
        const auto ref = SosReference::order( ws );
        ++numPlanes;
        numMismatches += segmentIntersectionTriPlaneOrder( ws ) != ref.taFirst;
        if ( ref.crossesB ? ref.nomDegree > cMaxDegTriPlane : ref.diffDegree > cMaxDegTriPlane )
            ++numAboveTriPlane;
        if ( !doTriangleSegmentIntersect( { ws[5], ws[6], ws[7], ws[0], ws[1] } ) )
            continue;
        ++numOrders;
        numMismatches += segmentIntersectionOrder( ws ) != ref.taFirst;
        if ( ref.nomDegree == 0 )
            ++numExact;
        else if ( ref.nomDegree <= cMaxDegTriTri )
            ++numDegenerate;
        else
            ++numAboveTriTri;
    }
    EXPECT_EQ( numMismatches, 0 );
    EXPECT_GT( numPlanes, cTrials / 10 );
    EXPECT_GT( numAboveTriPlane, cTrials / 200 );
    EXPECT_GT( numOrders, cTrials / 40 );
    EXPECT_GT( numExact, cTrials / 400 );
    EXPECT_GT( numDegenerate, cTrials / 50 );
    EXPECT_GT( numAboveTriTri, cTrials / 200 );
}

TEST( MRMesh, getToIntConverter )
{
    auto toInt = getToIntConverter( Box3d( {0,0,-1.0}, {0,0,1.0} ) );
    auto i0 = toInt( Vector3f{ 0,0,-1.f } );
    auto i1 = toInt( Vector3f{ 0,0, 1.f } );
    // check that sum and difference of any two points can be computed in integer without overflow
    EXPECT_LE( -i0.z, INT_MAX / 2 );
    EXPECT_LE(  i1.z, INT_MAX / 2 );
    EXPECT_GT( i1.z - i0.z, 0 );
}

} //namespace MR
