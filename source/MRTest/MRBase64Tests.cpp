#include <MRMesh/MRBase64.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, Base64 )
{
    EXPECT_EQ( decode64( "d29y" ), ( std::vector<std::uint8_t>{ 'w', 'o', 'r' } ) );
    EXPECT_EQ( decode64( "d28" ),  ( std::vector<std::uint8_t>{ 'w', 'o' } ) );
    EXPECT_EQ( decode64( "d28=" ), ( std::vector<std::uint8_t>{ 'w', 'o' } ) );
    EXPECT_EQ( decode64( "dw" ),   ( std::vector<std::uint8_t>{ 'w' } ) );
    EXPECT_EQ( decode64( "dw==" ), ( std::vector<std::uint8_t>{ 'w' } ) );
    EXPECT_EQ( decode64( "" ),     ( std::vector<std::uint8_t>{} ) );

    EXPECT_EQ( "d29y", encode64( (const std::uint8_t*)"wor", 3 ) );
    EXPECT_EQ( "d28=", encode64( (const std::uint8_t*)"wo", 2 ) );
    EXPECT_EQ( "dw==", encode64( (const std::uint8_t*)"w", 1 ) );
    EXPECT_EQ( "",     encode64( (const std::uint8_t*)"", 0 ) );
}

} //namespace MR
