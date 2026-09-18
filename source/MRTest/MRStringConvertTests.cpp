#include <MRMesh/MRStringConvert.h>
#include <gtest/gtest.h>

namespace MR
{

// "A", U+043B CYRILLIC SMALL LETTER EL (2 bytes), U+1F600 GRINNING FACE (4 bytes, outside the BMP)
static const std::string cUtf8 = "A\xD0\xBB\xF0\x9F\x98\x80";

TEST( MRMesh, Utf8Conversions )
{
    const auto u32 = utf8ToUtf32( cUtf8 );
    EXPECT_EQ( u32, ( std::u32string{ U'A', U'\x43B', U'\x1F600' } ) );
    EXPECT_EQ( utf32ToUtf8( u32 ), cUtf8 );

    const auto wide = utf8ToWide( cUtf8.c_str() );
    EXPECT_EQ( wide.size(), sizeof( wchar_t ) == 2 ? 4u : 3u ); // a surrogate pair in UTF-16
    EXPECT_EQ( wideToUtf8( wide.c_str() ), cUtf8 );

    EXPECT_TRUE( utf8ToWide( nullptr ).empty() );
    EXPECT_TRUE( utf8ToWide( "" ).empty() );
    EXPECT_TRUE( wideToUtf8( nullptr ).empty() );
    EXPECT_TRUE( wideToUtf8( L"" ).empty() );

    EXPECT_EQ( utf32ToUtf8( std::u32string{ U'\\xD800' } ), "\\xEF\\xBF\\xBD" );
    EXPECT_EQ( utf32ToUtf8( std::u32string{ char32_t( 0x110000 ) } ), "\\xEF\\xBF\\xBD" );
}

TEST( MRMesh, Utf8Substr )
{
    const std::string str = cUtf8 + "B";
    EXPECT_EQ( utf8substr( str.c_str(), 0, 4 ), str );
    EXPECT_EQ( utf8substr( str.c_str(), 0, 100 ), str );
    EXPECT_EQ( utf8substr( str.c_str(), 0, 0 ), "" );
    EXPECT_EQ( utf8substr( str.c_str(), 0, 1 ), "A" );
    EXPECT_EQ( utf8substr( str.c_str(), 1, 1 ), "\xD0\xBB" );
    EXPECT_EQ( utf8substr( str.c_str(), 2, 1 ), "\xF0\x9F\x98\x80" );
    EXPECT_EQ( utf8substr( str.c_str(), 3, 1 ), "B" );
    EXPECT_EQ( utf8substr( str.c_str(), 1, 2 ), "\xD0\xBB\xF0\x9F\x98\x80" );
    EXPECT_EQ( utf8substr( str.c_str(), 4, 1 ), "" );
    EXPECT_EQ( utf8substr( str.c_str(), 100, 1 ), "" );
}

} //namespace MR
