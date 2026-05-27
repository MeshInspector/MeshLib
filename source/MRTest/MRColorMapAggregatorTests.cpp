#include <MRMesh/MRColorMapAggregator.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRVector4.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, ColorMapAggregator )
{
    Color cWhite = Color::white();
    Color cRed = Color( Vector4i( 255, 0, 0, 128 ) );
    Color cGreen = Color( Vector4i( 0, 255, 0, 128 ) );

    FaceColorMapAggregator cma;
    cma.setDefaultColor( cWhite );

    int size = 5;
    FaceBitSet faces( size );
    faces.set( 1_f );
    faces.set( 2_f );
    cma.pushBack( { FaceColors( size, cRed ), faces } );

    faces.reset( 1_f );
    faces.set( 3_f );
    cma.pushBack( { FaceColors( size, cGreen ), faces } );

    cma.setMode( FaceColorMapAggregator::AggregateMode::Overlay );
    faces.set();
    FaceColors res = cma.aggregate( faces );

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == cRed );
    ASSERT_TRUE( res[2_f] == cGreen );
    ASSERT_TRUE( res[3_f] == cGreen );
    ASSERT_TRUE( res[4_f] == cWhite );


    cma.setMode( FaceColorMapAggregator::AggregateMode::Blending );
    res = cma.aggregate( faces );

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == Color( Vector4i( 255, 126, 126, 255 ) ) );
    ASSERT_TRUE( res[2_f] == Color( Vector4i( 126, 190, 62, 255 ) ) );
    ASSERT_TRUE( res[3_f] == Color( Vector4i( 126, 255, 126, 255 ) ) );
    ASSERT_TRUE( res[4_f] == cWhite );
}

} //namespace MR
