#include "MRRibbonIcons.h"
#include "MRMesh/MRImageLoad.h"
#include "MRImGuiImage.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <filesystem>

namespace MR
{

void RibbonIcons::load()
{
    auto& instance = instance_();
    instance.load_( IconType::RibbonItemIcon );
    instance.load_( IconType::ObjectTypeIcon );
}

void RibbonIcons::free()
{
    instance_().ribbonItemIconsMap_.clear();
}

const ImGuiImage* RibbonIcons::findByName( const std::string& name, float width, 
                                           ColorType colorType, IconType iconType )
{
    const auto& instance = instance_();
    const auto& map = iconType == IconType::RibbonItemIcon ?
        instance.ribbonItemIconsMap_ : instance.objectTypeIconsMap_;
    auto iconsIt = map.find( name );
    if ( iconsIt == map.end() )
        return nullptr;
    auto reqSize = int( instance.findRequiredSize_( width, iconType ) );
    if ( colorType == ColorType::Colored )
        return iconsIt->second[reqSize].colored.get();
    else
        return iconsIt->second[reqSize].white.get();
}

RibbonIcons& RibbonIcons::instance_()
{
    static RibbonIcons instance;
    return instance;
}

const char* RibbonIcons::sizeSubFolder_( Sizes sz )
{
    constexpr std::array<const char*, size_t( Sizes::Count )> folders =
    {
        "X0_5",
        "X1",
        "X3"
    };
    return folders[int( sz )];
}

RibbonIcons::Sizes RibbonIcons::findRequiredSize_( float width, IconType iconType ) const
{
    if ( iconType == IconType::RibbonItemIcon )
    {
        for ( int i = int( Sizes::MinRibbonItemIconSize ); i <= int( Sizes::MaxRibbonItemIconSize ); ++i )
        {
            float rate = float( loadedRibbonItemIconSizes_[i] ) / width;
            if ( rate > 0.95f ) // 5% upscaling is OK
                return Sizes( i );
        }
        return Sizes::MaxRibbonItemIconSize;
    }
    else
    {
        for ( int i = int( Sizes::MinObjectTypeIconSize ); i <= int( Sizes::MaxObjectTypeIconSize ); ++i )
        {
            float rate = float( loadedObjectTypeIconSizes_[i] ) / width;
            if ( rate > 0.95f ) // 5% upscaling is OK
                return Sizes( i );
        }
        return Sizes::MaxObjectTypeIconSize;
    }
}

void RibbonIcons::load_( IconType type )
{
    bool ribbonIconType = type == IconType::RibbonItemIcon;
    std::filesystem::path path = ribbonIconType ?
        GetResourcesDirectory() / "resource" / "icons" :
        GetResourcesDirectory() / "resource" / "object_icons";
    int minSize = ribbonIconType ? 
        int( Sizes::MinRibbonItemIconSize ) : int( Sizes::MinObjectTypeIconSize );
    int maxSize = ribbonIconType ?
        int( Sizes::MaxRibbonItemIconSize ) : int( Sizes::MaxObjectTypeIconSize );

    auto& map = ribbonIconType ? 
        ribbonItemIconsMap_ : objectTypeIconsMap_;
    auto& loadedSizes = ribbonIconType ? 
        loadedRibbonItemIconSizes_ : loadedObjectTypeIconSizes_;

    for ( int sz = minSize; sz <= maxSize; ++sz )
    {
        auto dirPath = path / sizeSubFolder_( Sizes( sz ) );

        std::error_code ec;
        if ( !std::filesystem::is_directory( dirPath, ec ) )
        {
            spdlog::error( "icons path {} is not directory", utf8string( dirPath ) );
            continue;
        }

        const std::filesystem::directory_iterator dirEnd;
        for ( auto it = std::filesystem::directory_iterator( dirPath, ec ); !ec && it != dirEnd; it.increment( ec ) )
        {
            if ( !it->is_regular_file( ec ) )
                continue;
            auto ext = it->path().extension().u8string();
            for ( auto& c : ext )
                c = ( char ) tolower( c );
            if ( ext != u8".png" )
                continue;
            auto image = ImageLoad::fromPng( it->path() );
            if ( !image.has_value() )
                continue;
            Icons icons;

            if ( ribbonIconType )
                icons.colored = std::make_unique<ImGuiImage>();

            icons.white = std::make_unique<ImGuiImage>();
            MeshTexture whiteTexture = { std::move( *image ) };
            if ( sz != int( Sizes::X0_5 ) )
                whiteTexture.filter = MeshTexture::FilterType::Linear;

            if ( ribbonIconType )
                icons.colored->update( whiteTexture );

            tbb::parallel_for( tbb::blocked_range<int>( 0, int( whiteTexture.pixels.size() ) ),
                               [&] ( const  tbb::blocked_range<int>& range )
            {
                for ( int i = range.begin(); i < range.end(); ++i )
                {
                    auto alpha = whiteTexture.pixels[i].a;
                    whiteTexture.pixels[i] = Color::white();
                    whiteTexture.pixels[i].a = alpha;
                }
            } );

            if ( loadedSizes[sz] == 0 )
                loadedSizes[sz] = whiteTexture.resolution.x;

            icons.white->update( std::move( whiteTexture ) );
            map[utf8string( it->path().stem() )][sz] = std::move( icons );
        }
    }
}

}