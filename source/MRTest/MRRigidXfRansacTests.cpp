#include <MRMesh/MRRigidXfRansac.h>
#include <MRMesh/MRVector.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRPointCloud.h>
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
        EXPECT_LT( distanceSq( ( *res )( p ), xf( p ) ), 1e-6f );
    EXPECT_EQ( countInliers( src, tgt, matches, *res, params.inlierDistance ), numCorrect );

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

TEST( MRMesh, RigidXfRansacOverlap )
{
    std::mt19937 gen( 2 );
    std::uniform_real_distribution<float> coord( -10, 10 );
    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 1.0f ), Vector3f( 20, -5, 7 ) );
    const AffineXf3f wrongXf( Matrix3f::rotation( Vector3f( 3, -1, 2 ).normalized(), 2.0f ), Vector3f( -30, 4, 1 ) );
    VertCoords src;
    PointCloud tgt;
    for ( int i = 0; i < 200; ++i )
    {
        src.push_back( Vector3f( coord( gen ), coord( gen ), coord( gen ) ) );
        tgt.addPoint( xf( src.back() ) );
    }
    // 20 correct matches, and 30 wrong ones consistent with another transformation, whose targets are few extra points
    std::vector<VertPair> matches;
    for ( int i = 0; i < 20; ++i )
        matches.emplace_back( VertId( i ), VertId( i ) );
    for ( int i = 20; i < 50; ++i )
        matches.emplace_back( VertId( i ), tgt.addPoint( wrongXf( src[VertId( i )] ) ) );

    RigidXfRansacParams params;
    params.inlierDistance = 1e-3f;
    // most inlier matches belong to the wrong transformation
    auto res = findRigidXfRansac( src, tgt.points, matches, params );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( countInliers( src, tgt.points, matches, *res, params.inlierDistance ), 30 );

    // but the right one moves all source points onto the target cloud
    params.overlapSrcPoints = &src;
    params.overlapTgtCloud = &tgt;
    res = findRigidXfRansac( src, tgt.points, matches, params );
    ASSERT_TRUE( res.has_value() );
    for ( const auto & p : src )
        EXPECT_LT( distanceSq( ( *res )( p ), xf( p ) ), 1e-6f );

    // and its inverse moves most target points onto the source cloud
    PointCloud srcCloud;
    for ( const auto & p : src )
        srcCloud.addPoint( p );
    params.overlapSrcPoints = nullptr;
    params.overlapTgtCloud = nullptr;
    params.overlapTgtPoints = &tgt.points;
    params.overlapSrcCloud = &srcCloud;
    res = findRigidXfRansac( src, tgt.points, matches, params );
    ASSERT_TRUE( res.has_value() );
    for ( const auto & p : src )
        EXPECT_LT( distanceSq( ( *res )( p ), xf( p ) ), 1e-6f );
}

} //namespace MR
