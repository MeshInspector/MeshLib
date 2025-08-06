#include "MRRibbonIcons.h"
#include "MRImGuiImage.h"
#include "MRViewer.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

#include "MRIOExtras/MRPng.h"

namespace MR
{

void RibbonIcons::load()
{
    MR_TIMER;
    auto& instance = instance_();
    instance.load_( IconType::RibbonItemIcon );
    instance.load_( IconType::ObjectTypeIcon );
    instance.load_( IconType::IndependentIcons );
    instance.load_( IconType::Logos );
}

void RibbonIcons::free()
{
    for ( auto& curData : instance_().data_ )
    {
        curData.map.clear();
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
    return instance.findRequiredSize_( iconsIt->second, width, colorType, iconType );
}

RibbonIcons::RibbonIcons()
{
    data_[size_t( IconType::RibbonItemIcon )] = {
        SystemPath::getResourcesDirectory() / "resource" / "icons",
        std::make_pair( Sizes::X0_5, Sizes::X3 ),
        IconTypeData::AvailableColor::White | IconTypeData::AvailableColor::Colored,
    };

    data_[size_t( IconType::ObjectTypeIcon )] = {
        SystemPath::getResourcesDirectory() / "resource" / "object_icons",
        std::make_pair( Sizes::X1, Sizes::X3 ),
        IconTypeData::AvailableColor::White,
    };

    data_[size_t( IconType::IndependentIcons )] = {
        SystemPath::getResourcesDirectory() / "resource" / "independent_icons",
        std::make_pair( Sizes::X1, Sizes::X3 ),
        IconTypeData::AvailableColor::White,
    };

    data_[size_t( IconType::Logos )] = {
        SystemPath::getResourcesDirectory() / "resource" / "logos",
        std::make_pair( Sizes::X1, Sizes::X3 ),
        IconTypeData::AvailableColor::Colored,
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

const ImGuiImage* RibbonIcons::findRequiredSize_( const SizedIcons& icons, float width, ColorType colorType, IconType iconType ) const
{
    const auto& curData = data_[size_t( iconType )];

    int maxSize = -1;
    const ImGuiImage* maxIcon{ nullptr };
    for ( int i = int( curData.minMaxSizes.first ); i <= int( curData.minMaxSizes.second ); ++i )
    {
        const auto& icon = colorType == ColorType::White ? icons[i].white : icons[i].colored;
        if ( !icon )
            continue;
        if ( i > maxSize )
        {
            maxSize = i;
            maxIcon = icon.get();
        }
        float rate = float( icon->getImageWidth() ) / width;
        if ( rate > 0.95f ) // 5% upscaling is OK
            return maxIcon;
    }
    return maxIcon;
}

void RibbonIcons::load_( IconType type )
{
    size_t num = static_cast< size_t >( type );
    auto& currentData = data_[num];

    bool coloredIcons = bool( currentData.availableColor & IconTypeData::AvailableColor::Colored );
    bool whiteIcons = bool( currentData.availableColor & IconTypeData::AvailableColor::White );

    std::filesystem::path path = currentData.pathDirectory;
    int minSize = static_cast< int >( currentData.minMaxSizes.first );
    int maxSize = static_cast< int >( currentData.minMaxSizes.second );

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

            MeshTexture texture{ std::move( *image ) };
            if ( sz != int( Sizes::X0_5 ) )
                texture.filter = FilterType::Linear;

            Icons icons;
            if ( coloredIcons )
            {
                icons.colored = std::make_unique<ImGuiImage>();
                icons.colored->update( texture );
            }
            if ( whiteIcons )
            {
                icons.white = std::make_unique<ImGuiImage>();
                tbb::parallel_for( tbb::blocked_range<int>( 0, int( texture.pixels.size() ) ),
                                   [&] ( const  tbb::blocked_range<int>& range )
                {
                    for ( int i = range.begin(); i < range.end(); ++i )
                    {
                        auto alpha = texture.pixels[i].a;
                        texture.pixels[i] = Color::white();
                        texture.pixels[i].a = alpha;
                    }
                } );
                icons.white->update( std::move( texture ) );
            }
            currentData.map[utf8string( entry.path().stem() )][sz] = std::move( icons );
        }
    }
}

}
