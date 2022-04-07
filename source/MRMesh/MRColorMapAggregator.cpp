#include "MRColorMapAggregator.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRColorMapAggregator, Face )
{
    Color cWhite = Color::white();
    Color cRed = Color(Vector4i(255, 0, 0, 128));
    Color cGreen = Color( Vector4i( 0, 255, 0, 128 ) );

    FaceColorMapAggregator cma;
    cma.setDefaultColor( cWhite );
    
    int size = 5;
    FaceBitSet faces;
    faces.resize( 5, true );
    cma.setColorMap( 2, Vector<Color, FaceId>( size, cRed ), FaceBitSet( std::string( "00110" ) ) );
    cma.setColorMap( 4, Vector<Color, FaceId>( size, cGreen ), FaceBitSet( std::string( "01100" ) ) );
    cma.setMode( FaceColorMapAggregator::AggregateMode::Overlay );
    Vector<Color, FaceId> res = cma.aggregate();

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == cRed );
    ASSERT_TRUE( res[2_f] == cGreen );
    ASSERT_TRUE( res[3_f] == cGreen );
    ASSERT_TRUE( res[4_f] == cWhite );


    cma.setMode( FaceColorMapAggregator::AggregateMode::Blending );
    res = cma.aggregate();

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == Color( Vector4i( 255, 126, 126, 255 ) ) );
    ASSERT_TRUE( res[2_f] == Color( Vector4i( 126, 190, 62, 255 ) ) );
    ASSERT_TRUE( res[3_f] == Color( Vector4i( 126, 255, 126, 255 ) ) );
    ASSERT_TRUE( res[4_f] == cWhite );
}


}
