#include <MRIOExtras/MRSvg.h>
#ifndef MRIOEXTRAS_NO_XML
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace MR
{

// a file with symbols outside of any single code page in its name must be opened on Windows as well
TEST( MRIOExtras, SvgUnicodeFileName )
{
    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const auto path = folder / std::filesystem::path( u8"\u0442\u0435\u0441\u0442 \u6587\u4ef6 \u00e9.svg" );
    std::ofstream( path, std::ofstream::binary )
        << R"(<svg xmlns="http://www.w3.org/2000/svg"><line x1="0" y1="0" x2="10" y2="5"/></svg>)";

    const auto loaded = LinesLoad::fromSvg( path );
    ASSERT_TRUE( loaded ) << loaded.error();
    EXPECT_EQ( loaded->topology.numValidVerts(), 2 );

    const auto absent = LinesLoad::fromSvg( folder / "absent.svg" );
    ASSERT_FALSE( absent );
    EXPECT_TRUE( absent.error().starts_with( "Cannot open file for reading" ) ) << absent.error();
}

} // namespace MR
#endif
