#include <MRMesh/MRAABBTreePoints.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBall.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRDirMax.h>
#include <MRMesh/MRFewSmallest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshToPointCloud.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointsInBall.h>
#include <MRMesh/MRPointsInBox.h>
#include <MRMesh/MRPointsProject.h>
#include <MRMesh/MRVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace MR
{

TEST( MRMesh, AABBTreePoints )
{
    PointCloud spherePC = meshToPointCloud( makeUVSphere( 1, 8, 8 ) );
    AABBTreePoints tree( spherePC );
    EXPECT_EQ( tree.orderedPoints().size(), spherePC.validPoints.count() );

    Box3f box;
    for ( auto v : spherePC.validPoints )
        box.include( spherePC.points[v] );

    EXPECT_EQ( tree.getBoundingBox(), box );
    ASSERT_FALSE( tree.simdNodes().empty() );
    EXPECT_EQ( tree.rootRef(), 0 );
    EXPECT_NE( tree.simdNodes()[0].child[1], 0 );

    EXPECT_FALSE( tree.empty() );
    auto m = std::move( tree );
    EXPECT_FALSE( m.empty() );
}

TEST( MRMesh, AABBTreePointsFromMesh )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTreePoints tree( sphere );
    EXPECT_EQ( tree.orderedPoints().size(), sphere.topology.numValidVerts() );

    Box3f box;
    for ( auto v : sphere.topology.getValidVerts() )
        box.include( sphere.points[v] );

    EXPECT_EQ( tree.getBoundingBox(), box );
    ASSERT_FALSE( tree.simdNodes().empty() );
    EXPECT_NE( tree.simdNodes()[0].child[1], 0 );
}

namespace
{

/// checks that every leaf is referenced once, the children follow their parent, and every child box is exactly the box of its points
void checkTreeStructure( const AABBTreePoints & tree, const VertCoords & points, const VertBitSet * valid )
{
    const auto & op = tree.orderedPoints();
    ASSERT_EQ( op.size(), valid ? valid->count() : points.size() );
    VertBitSet seen( points.size() );
    for ( const auto & p : op )
    {
        EXPECT_FALSE( seen.test_set( p.id ) );
        EXPECT_EQ( p.coord, points[p.id] );
    }
    if ( op.empty() )
        return;

    // the box of all points in the subtree of the node with given reference
    std::vector<int> leafRefs( tree.numLeaves(), 0 );
    auto subtreeBox = [&] ( int ref, int depth, auto && self ) -> Box3f
    {
        EXPECT_LE( depth, AABBTreePoints::MaxDepth );
        Box3f res;
        if ( ref < 0 )
        {
            ++leafRefs[~ref];
            const auto [first, last] = tree.leafPointRange( ~ref );
            EXPECT_TRUE( last - first == AABBTreePoints::MaxNumPointsInLeaf || last == int( op.size() ) );
            for ( int i = first; i < last; ++i )
                res.include( op[i].coord );
            return res;
        }
        const auto & node = tree.simdNodes()[ref];
        EXPECT_NE( node.child[0], 0 );
        EXPECT_NE( node.child[1], 0 );
        for ( int i = 0; i < AABBTreePoints::MaxChildren; ++i )
        {
            if ( !node.child[i] )
            {
                EXPECT_FALSE( node.get( i ).valid() );
                continue;
            }
            EXPECT_TRUE( node.child[i] < 0 || node.child[i] > ref );
            const auto b = self( node.child[i], depth + 1, self );
            EXPECT_EQ( node.get( i ), b );
            res.include( b );
        }
        return res;
    };
    EXPECT_EQ( subtreeBox( tree.rootRef(), 0, subtreeBox ), tree.getBoundingBox() );
    for ( int c : leafRefs )
        EXPECT_EQ( c, 1 );
}

VertCoords randomPoints( int n, unsigned seed )
{
    std::mt19937 rng( seed );
    std::uniform_real_distribution<float> u( -1, 1 );
    VertCoords res;
    res.resize( n );
    for ( auto & p : res )
        p = { u( rng ), u( rng ), u( rng ) };
    return res;
}

} // anonymous namespace

TEST( MRMesh, AABBTreePointsStructure )
{
    for ( int n : { 1, 15, 16, 17, 32, 33, 48, 63, 64, 65, 100, 257, 1000, 4097, 12345, 70000 } )
    {
        const auto points = randomPoints( n, n );
        checkTreeStructure( AABBTreePoints( points ), points, nullptr );

        VertBitSet valid( n );
        for ( int i = 0; i < n; i += 3 )
            valid.set( VertId( i ) );
        checkTreeStructure( AABBTreePoints( points, &valid ), points, &valid );
    }
    const VertCoords none;
    AABBTreePoints empty( none );
    EXPECT_TRUE( empty.empty() );
    EXPECT_FALSE( empty.getBoundingBox().valid() );
}

TEST( MRMesh, AABBTreePointsRefit )
{
    for ( int n : { 10, 100, 5000 } )
    {
        auto points = randomPoints( n, 7 );
        AABBTreePoints tree( points );
        // move a part of the points
        VertBitSet changed( n );
        for ( int i = 0; i < n; i += 7 )
        {
            changed.set( VertId( i ) );
            points[VertId( i )] = 3.0f * points[VertId( i )] + Vector3f( 1, 0, 0 );
        }
        tree.refit( points, &changed );
        checkTreeStructure( tree, points, nullptr );
        // move all points
        for ( auto & p : points )
            p *= 0.5f;
        tree.refit( points, nullptr );
        checkTreeStructure( tree, points, nullptr );
    }
}

/// the searches in the tree find the same as brute force
TEST( MRMesh, AABBTreePointsSearches )
{
    PointCloud pc;
    pc.points = randomPoints( 3000, 1 );
    pc.validPoints.resize( pc.points.size(), true );
    for ( int i = 0; i < 3000; i += 5 )
        pc.validPoints.reset( VertId( i ) );
    const auto & tree = pc.getAABBTree();

    VertBitSet region( pc.points.size() );
    for ( int i = 0; i < 3000; i += 2 )
        region.set( VertId( i ) );
    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.7f ), Vector3f( 0.1f, -0.2f, 0.3f ) );

    std::mt19937 rng( 2 );
    std::uniform_real_distribution<float> u( -1.5f, 1.5f );
    for ( int q = 0; q < 300; ++q )
    {
        const Vector3f pt( u( rng ), u( rng ), u( rng ) );
        for ( int mode = 0; mode < 3; ++mode )
        {
            const AffineXf3f * pxf = mode == 1 ? &xf : nullptr;
            const VertBitSet * reg = mode == 2 ? &region : nullptr;
            float best = FLT_MAX;
            for ( auto v : pc.validPoints )
                if ( !reg || reg->test( v ) )
                    best = std::min( best, ( ( pxf ? ( *pxf )( pc.points[v] ) : pc.points[v] ) - pt ).lengthSq() );
            const auto res = findProjectionOnPoints( pt, tree, FLT_MAX, pxf, 0, reg );
            EXPECT_EQ( res.distSq, best );
            ASSERT_TRUE( res.vId.valid() );
            EXPECT_EQ( ( ( pxf ? ( *pxf )( pc.points[res.vId] ) : pc.points[res.vId] ) - pt ).lengthSq(), best );
        }

        // the few closest points
        FewSmallest<PointsProjectionResult> few( 5 );
        findFewClosestPoints( pt, pc, few );
        std::vector<float> got, expected;
        for ( const auto & r : few.get() )
            got.push_back( r.distSq );
        for ( auto v : pc.validPoints )
            expected.push_back( ( pc.points[v] - pt ).lengthSq() );
        std::sort( got.begin(), got.end() );
        std::sort( expected.begin(), expected.end() );
        expected.resize( 5 );
        EXPECT_EQ( got, expected );

        // the points in a ball and in a box
        const float r = 0.3f;
        std::vector<int> inBall, inBallExp, inBox, inBoxExp;
        findPointsInBall( tree, Ball3f( pt, r * r ), [&] ( const PointsProjectionResult & p, const Vector3f &, Ball3f & )
        {
            inBall.push_back( p.vId );
            return Processing::Continue;
        } );
        const Box3f box( pt - Vector3f::diagonal( r ), pt + Vector3f::diagonal( r ) );
        findPointsInBox( tree, box, [&] ( VertId v, const Vector3f & ) { inBox.push_back( v ); } );
        for ( auto v : pc.validPoints )
        {
            if ( ( pc.points[v] - pt ).lengthSq() <= r * r )
                inBallExp.push_back( v );
            if ( box.contains( pc.points[v] ) )
                inBoxExp.push_back( v );
        }
        std::sort( inBall.begin(), inBall.end() );
        std::sort( inBox.begin(), inBox.end() );
        EXPECT_EQ( inBall, inBallExp );
        EXPECT_EQ( inBox, inBoxExp );

        // the extreme point
        const Vector3f dir( u( rng ), u( rng ), u( rng ) );
        float maxProj = -FLT_MAX;
        for ( auto v : pc.validPoints )
            maxProj = std::max( maxProj, dot( pc.points[v], dir ) );
        const auto v = findDirMax( dir, tree );
        ASSERT_TRUE( v.valid() );
        EXPECT_EQ( dot( pc.points[v], dir ), maxProj );
    }
}

TEST( MRMesh, AABBTreePointsSearchOptions )
{
    PointCloud pc;
    pc.points = randomPoints( 3000, 4 );
    pc.validPoints.resize( pc.points.size(), true );
    const auto & tree = pc.getAABBTree();
    VertBitSet region( pc.points.size() );
    for ( int i = 0; i < 3000; i += 3 )
        region.set( VertId( i ) );
    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 3, -1, 2 ).normalized(), 1.1f ), Vector3f( -0.2f, 0.1f, 0.4f ) );

    std::mt19937 rng( 5 );
    std::uniform_real_distribution<float> u( -1.5f, 1.5f );
    for ( int q = 0; q < 200; ++q )
    {
        const Vector3f pt( u( rng ), u( rng ), u( rng ) );
        for ( const AffineXf3f * pxf : { (const AffineXf3f *)nullptr, &xf } )
        {
            auto pos = [&] ( VertId v ) { return pxf ? ( *pxf )( pc.points[v] ) : pc.points[v]; };

            // the points in a ball with the transformation
            const float r = 0.3f;
            std::vector<int> inBall, inBallExp;
            findPointsInBall( tree, Ball3f( pt, r * r ), [&] ( const PointsProjectionResult & p, const Vector3f & xfPos, Ball3f & )
            {
                EXPECT_EQ( xfPos, pos( p.vId ) );
                EXPECT_EQ( p.distSq, ( xfPos - pt ).lengthSq() );
                inBall.push_back( p.vId );
                return Processing::Continue;
            }, pxf );
            for ( auto v : pc.validPoints )
                if ( ( pos( v ) - pt ).lengthSq() <= r * r )
                    inBallExp.push_back( v );
            std::sort( inBall.begin(), inBall.end() );
            EXPECT_EQ( inBall, inBallExp );

            // the closest point by a ball shrinking to every found point
            float bestBall = FLT_MAX;
            findPointsInBall( tree, Ball3f( pt, 1.0f ), [&] ( const PointsProjectionResult & p, const Vector3f &, Ball3f & ball )
            {
                EXPECT_LE( p.distSq, ball.radiusSq );
                bestBall = std::min( bestBall, p.distSq );
                ball.radiusSq = p.distSq;
                return Processing::Continue;
            }, pxf );
            float best = FLT_MAX;
            for ( auto v : pc.validPoints )
                best = std::min( best, ( pos( v ) - pt ).lengthSq() );
            if ( best <= 1.0f )
                EXPECT_EQ( bestBall, best );
            else
                EXPECT_EQ( bestBall, FLT_MAX );

            // the projection skipping the points out of the region by the callback
            float bestSkip = FLT_MAX;
            for ( auto v : region )
                bestSkip = std::min( bestSkip, ( pos( v ) - pt ).lengthSq() );
            const auto res = findProjectionOnPoints( pt, tree, FLT_MAX, pxf, 0, nullptr, [&] ( VertId v ) { return !region.test( v ); } );
            ASSERT_TRUE( res.vId.valid() );
            EXPECT_TRUE( region.test( res.vId ) );
            EXPECT_EQ( res.distSq, bestSkip );
        }

        // the extreme point in the region
        const Vector3f dir( u( rng ), u( rng ), u( rng ) );
        float maxProj = -FLT_MAX;
        for ( auto v : region )
            maxProj = std::max( maxProj, dot( pc.points[v], dir ) );
        const auto v = findDirMax( dir, tree, &region );
        ASSERT_TRUE( v.valid() );
        EXPECT_TRUE( region.test( v ) );
        EXPECT_EQ( dot( pc.points[v], dir ), maxProj );
    }
}

TEST( MRMesh, AABBTreePointsNonFiniteQueries )
{
    PointCloud pc;
    pc.points = randomPoints( 3000, 3 );
    pc.validPoints.resize( pc.points.size(), true );
    const auto & tree = pc.getAABBTree();
    // the unused slots of a node refer to 0, the root, and must never be visited
    bool hasUnusedSlot = false;
    for ( const auto & n : tree.simdNodes() )
        hasUnusedSlot = hasUnusedSlot || n.child[AABBTreePoints::MaxChildren - 1] == 0;
    ASSERT_TRUE( hasUnusedSlot );

    const float nan = std::numeric_limits<float>::quiet_NaN();
    for ( const Vector3f pt : { Vector3f( nan, 0, 0 ), Vector3f( 0, 0, nan ), Vector3f::diagonal( nan ) } )
    {
        EXPECT_FALSE( findProjectionOnPoints( pt, tree ).vId.valid() );
        FewSmallest<PointsProjectionResult> few( 5 );
        findFewClosestPoints( pt, pc, few );
        EXPECT_TRUE( few.empty() );
    }

    const float inf = std::numeric_limits<float>::infinity();
    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.7f ), Vector3f( 0.1f, -0.2f, 0.3f ) );
    for ( const AffineXf3f * pxf : { (const AffineXf3f *)nullptr, &xf } )
    {
        size_t num = 0;
        findPointsInBox( tree, Box3f( Vector3f::diagonal( -inf ), Vector3f::diagonal( inf ) ), [&] ( VertId, const Vector3f & ) { ++num; }, pxf );
        EXPECT_EQ( num, pc.points.size() );
    }
}

} // namespace MR
