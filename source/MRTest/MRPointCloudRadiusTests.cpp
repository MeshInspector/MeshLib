#include <MRMesh/MRPointCloudRadius.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, PointCloudDilateErodeRegion )
{
    // 10 points on a line with unit spacing
    PointCloud pc;
    for ( int i = 0; i < 10; ++i )
        pc.addPoint( Vector3f( float( i ), 0, 0 ) );

    auto range = []( int first, int last )
    {
        VertBitSet res( 10 );
        for ( int i = first; i <= last; ++i )
            res.set( VertId( i ) );
        return res;
    };

    auto region = range( 2, 7 );
    EXPECT_TRUE( erodeRegion( pc, region, 0.5f ) );
    EXPECT_EQ( region, range( 2, 7 ) );
    EXPECT_TRUE( erodeRegion( pc, region, 1.5f ) );
    EXPECT_EQ( region, range( 3, 6 ) );
    EXPECT_TRUE( dilateRegion( pc, region, 1.5f ) );
    EXPECT_EQ( region, range( 2, 7 ) );
    EXPECT_TRUE( dilateRegion( pc, region, 1.5f ) );
    EXPECT_EQ( region, range( 1, 8 ) );

    // nothing to erode from if all points are in the region
    region = range( 0, 9 );
    EXPECT_TRUE( erodeRegion( pc, region, 1.5f ) );
    EXPECT_EQ( region, range( 0, 9 ) );

    // distances are measured after transformation
    const auto xf = AffineXf3f::linear( Matrix3f::scale( 2.0f ) );
    region = range( 2, 7 );
    EXPECT_TRUE( erodeRegion( pc, region, 1.5f, {}, &xf ) );
    EXPECT_EQ( region, range( 2, 7 ) );
    EXPECT_TRUE( erodeRegion( pc, region, 2.5f, {}, &xf ) );
    EXPECT_EQ( region, range( 3, 6 ) );

    // removed points do not erode the region
    pc.validPoints.reset( VertId( 8 ) );
    pc.invalidateCaches();
    region = range( 2, 7 );
    EXPECT_TRUE( erodeRegion( pc, region, 1.5f ) );
    EXPECT_EQ( region, range( 3, 7 ) );
}

} //namespace MR
