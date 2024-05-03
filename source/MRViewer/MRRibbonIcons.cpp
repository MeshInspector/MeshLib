#include "MRRibbonIcons.h"
#include "MRImGuiImage.h"
#include "MRViewer.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRDirectory.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

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
    for ( auto& curData : instance_().data_ )
    {
        curData.map.clear();
        curData.loadSize = std::array<int, size_t( Sizes::Count )>();
    }
}

const ImGuiImage* RibbonIcons::findByName( const std::string& name, float width, 
                                           ColorType colorType, IconType iconType )
{
    const auto& instance = instance_();
    const auto& map = instance.data_[size_t(iconType)].map;
    auto iconsIt = map.find( name );
    if ( iconsIt == map.end() )
        return nullptr;
    auto reqSize = int( instance.findRequiredSize_( width, iconType ) );
    if ( colorType == ColorType::Colored )
        return iconsIt->second[reqSize].colored.get();
    else
        return iconsIt->second[reqSize].white.get();
}

RibbonIcons::RibbonIcons()
{
    data_[size_t( IconType::RibbonItemIcon )] = {
        GetResourcesDirectory() / "resource" / "icons",
        std::make_pair( Sizes::X0_5, Sizes::X3 ),
        true,
    };

    data_[size_t( IconType::ObjectTypeIcon )] = {
        GetResourcesDirectory() / "resource" / "object_icons",
        std::make_pair( Sizes::X1, Sizes::X3 ),
        false,
    };

    data_[size_t( IconType::IndependentIcons )] = {
        GetResourcesDirectory() / "resource" / "independent_icons",
        std::make_pair( Sizes::X1, Sizes::X3 ),
        false,
    };
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
        "X0_75",
        "X1",
        "X3"
    };
    return folders[int( sz )];
}

RibbonIcons::Sizes RibbonIcons::findRequiredSize_( float width, IconType iconType ) const
{
    const auto& curData = data_[size_t( iconType )];

    for ( int i = int( curData.minMaxSizes.first ); i <= int( curData.minMaxSizes.second ); ++i )
    {
        float rate = float( curData.loadSize[i] ) / width;
        if ( rate > 0.95f ) // 5% upscaling is OK
            return Sizes( i );
    }
    return curData.minMaxSizes.second;
}

void RibbonIcons::load_( IconType type )
{
    size_t num = static_cast<size_t>( type );
    auto& currentData = data_[num];

    bool coloredIcons = currentData.needLoadColored;

    std::filesystem::path path = currentData.pathDirectory;
    int minSize = static_cast< int >( currentData.minMaxSizes.first );
    int maxSize = static_cast< int >( currentData.minMaxSizes.second );

    //auto& map = currentData.map;
    auto& loadedSizes = currentData.loadSize;

    for ( int sz = minSize; sz <= maxSize; ++sz )
    {
        auto dirPath = path / sizeSubFolder_( Sizes( sz ) );

        std::error_code ec;
        if ( !std::filesystem::is_directory( dirPath, ec ) )
        {
            spdlog::error( "icons path {} is not directory", utf8string( dirPath ) );
            continue;
        }

        for ( auto entry : Directory{ dirPath, ec } )
        {
            if ( !entry.is_regular_file( ec ) )
                continue;
            auto ext = entry.path().extension().u8string();
            for ( auto& c : ext )
                c = ( char )tolower( c );
            if ( ext != u8".png" )
                continue;
            auto image = ImageLoad::fromPng( entry.path() );
            if ( !image.has_value() )
                continue;
            Icons icons;

            if ( coloredIcons )
                icons.colored = std::make_unique<ImGuiImage>();

            icons.white = std::make_unique<ImGuiImage>();
            MeshTexture whiteTexture = { std::move( *image ) };
            if ( sz != int( Sizes::X0_5 ) )
                whiteTexture.filter = FilterType::Linear;

            if ( coloredIcons )
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
            currentData.map[utf8string( entry.path().stem() )][sz] = std::move( icons );
        }
    }
}

}
