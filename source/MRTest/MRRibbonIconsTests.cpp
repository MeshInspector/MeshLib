#include <MRViewer/MRRibbonIcons.h>
#include <MRIOExtras/MRPng.h>
#include <MRMesh/MRImage.h>
#include <MRMesh/MRSystemPath.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRMesh/MRZip.h>
#include <gtest/gtest.h>

namespace MR
{

// resource/icons.zip replaces the per-type icon folders
TEST( MRViewer, RibbonIconsFromZip )
{
    UniqueTemporaryFolder tmp;
    ASSERT_TRUE( bool( tmp ) );

    const auto iconsDir = tmp / "icons_src";
    const Image image{ .pixels = std::vector<Color>( 4, Color::red() ), .resolution = { 2, 2 } };
    for ( const char* file : { "icons/X1/Tool.png", "icons/X3/Tool.png", "object_icons/X1/mesh.png" } )
    {
        std::error_code ec;
        std::filesystem::create_directories( ( iconsDir / file ).parent_path(), ec );
        ASSERT_TRUE( ImageSave::toPng( image, iconsDir / file ).has_value() );
    }
    const auto resourcesDir = tmp / "resources";
    std::error_code ec;
    std::filesystem::create_directories( resourcesDir / "resource", ec );
    ASSERT_TRUE( compressZip( resourcesDir / "resource" / "icons.zip", iconsDir ).has_value() );

    const auto oldResourcesDir = SystemPath::getResourcesDirectory();
    SystemPath::overrideDirectory( SystemPath::Directory::Resources, resourcesDir );
    RibbonIcons::free();
    EXPECT_TRUE( RibbonIcons::findByName( "Tool", 32, RibbonIcons::ColorType::White, RibbonIcons::IconType::RibbonItemIcon ) );
    EXPECT_TRUE( RibbonIcons::findByName( "Tool", 32, RibbonIcons::ColorType::Colored, RibbonIcons::IconType::RibbonItemIcon ) );
    EXPECT_TRUE( RibbonIcons::findByName( "mesh", 32, RibbonIcons::ColorType::White, RibbonIcons::IconType::ObjectTypeIcon ) );
    EXPECT_FALSE( RibbonIcons::findByName( "mesh", 32, RibbonIcons::ColorType::White, RibbonIcons::IconType::RibbonItemIcon ) );
    EXPECT_FALSE( RibbonIcons::findByName( "Tool", 32, RibbonIcons::ColorType::White, RibbonIcons::IconType::ObjectTypeIcon ) );
    RibbonIcons::free();
    SystemPath::overrideDirectory( SystemPath::Directory::Resources, oldResourcesDir );
}

} //namespace MR
