#include <MRMesh/MRPrecisePredicates2.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <MRMesh/MRInSphere.h>
#include <MRMesh/MRBox.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>

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

    // maximal magnitudes as if from getToIntConverter: no overflow in internal computations
    constexpr int H = 1'000'000'000;
    const Vector3i a1{ -H, -H, 0 }, b1{ H, -H, 0 }, c1{ -H, H, 0 };
    // circumcircle center (0,0,0), squared radius 2*H^2; with rSq = 3*H^2 sphere center is (0,0,H)
    const auto rSq = 3 * sqr( std::int64_t( H ) );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ 0, 0, 2 * H }, rSq ), In );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ 0, 0, -1 }, rSq ), In );
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ H, H, 2 * H }, rSq ), On ); // exactly on the sphere
    EXPECT_EQ( inSphere( a1, b1, c1, Vector3i{ 2 * H, 0, 0 }, rSq ), Out );
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
