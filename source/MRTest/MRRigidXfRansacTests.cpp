#include <MRMesh/MRRigidXfRansac.h>
#include <MRMesh/MRVector.h>
#include <MRMesh/MRMatrix3.h>
#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <random>

namespace MR
{

TEST( MRMesh, RigidXfRansac )
{
    std::mt19937 gen( 1 );
    std::uniform_real_distribution<float> coord( -10, 10 );
    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 1.0f ), Vector3f( 20, -5, 7 ) );
    VertCoords src, tgt;
    for ( int i = 0; i < 200; ++i )
    {
        src.push_back( Vector3f( coord( gen ), coord( gen ), coord( gen ) ) );
        tgt.push_back( xf( src.back() ) );
    }
    // only 20% of the matches are correct
    std::vector<VertPair> matches;
    size_t numCorrect = 0;
    for ( int i = 0; i < 200; ++i )
    {
        const VertId t( i < 40 ? i : int( gen() % 200 ) );
        matches.emplace_back( VertId( i ), t );
        numCorrect += t == VertId( i );
    }

    RigidXfRansacParams params;
    params.inlierDistance = 1e-3f;
    auto res = findRigidXfRansac( src, tgt, matches, params );
    ASSERT_TRUE( res.has_value() );
    for ( const auto & p : src )
        EXPECT_LT( distance( ( *res )( p ), xf( p ) ), 1e-3f );
    size_t numInliers = 0;
    for ( const auto & m : matches )
        numInliers += distance( ( *res )( src[m.first] ), tgt[m.second] ) <= params.inlierDistance;
    EXPECT_EQ( numInliers, numCorrect );

    // the same result with a single thread
    {
        tbb::global_control singleThread( tbb::global_control::max_allowed_parallelism, 1 );
        auto res1 = findRigidXfRansac( src, tgt, matches, params );
        ASSERT_TRUE( res1.has_value() );
        EXPECT_EQ( *res1, *res );
    }

    matches.resize( 2 );
    EXPECT_FALSE( findRigidXfRansac( src, tgt, matches, params ).has_value() );
}

} //namespace MR
