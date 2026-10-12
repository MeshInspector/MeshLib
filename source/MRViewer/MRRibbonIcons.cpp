#include "MRRibbonIcons.h"
#include "MRImGuiImage.h"
#include "MRViewer.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRZip.h"
#include "MRPch/MRSpdlog.h"

#include "MRIOExtras/MRPng.h"

namespace MR
{

namespace
{

// std::istream over a buffer in memory
struct MemoryStreamBuf : std::streambuf
{
    MemoryStreamBuf( std::vector<char>& data ) { setg( data.data(), data.data(), data.data() + data.size() ); }
};

bool isPng( const std::filesystem::path& file )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = ( char )tolower( c );
    return ext == u8".png";
}

} // anonymous namespace

void RibbonIcons::load()
{
    MR_TIMER;
    auto& instance = instance_();
    instance.load_( IconType::RibbonItemIcon );
    instance.load_( IconType::ObjectTypeIcon );
    instance.load_( IconType::IndependentIcons );
    // without icons.zip, IconType::Logos is loaded lazily on first request
}

void RibbonIcons::free()
{
    auto& instance = instance_();
    for ( auto& curData : instance.data_ )
    {
        curData.map.clear();
        curData.loaded = false;
    }
    instance.zipTried_ = false;
}

const ImGuiImage* RibbonIcons::findByName( const std::string& name, float width, 
                                           ColorType colorType, IconType iconType )
{
    auto& instance = instance_();
    auto& data = instance.data_[size_t(iconType)];
    if ( !data.loaded )
        instance.load_( iconType );
    const auto& map = data.map;
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

void RibbonIcons::addIcon_( IconTypeData& data, int sz, const std::string& name, Image&& image )
{
    MeshTexture texture{ std::move( image ) };
    if ( sz != int( Sizes::X0_5 ) )
        texture.filter = FilterType::Linear;

    Icons icons;
    if ( bool( data.availableColor & IconTypeData::AvailableColor::Colored ) )
    {
        icons.colored = std::make_unique<ImGuiImage>();
        icons.colored->update( texture );
    }
    if ( bool( data.availableColor & IconTypeData::AvailableColor::White ) )
    {
        icons.white = std::make_unique<ImGuiImage>();
        ParallelFor( texture.pixels, [&] ( size_t i )
        {
            auto alpha = texture.pixels[i].a;
            texture.pixels[i] = Color::white();
            texture.pixels[i].a = alpha;
        } );
        icons.white->update( std::move( texture ) );
    }
    data.map[name][sz] = std::move( icons );
}

bool RibbonIcons::loadZip_()
{
    const auto zipPath = SystemPath::getResourcesDirectory() / "resource" / "icons.zip";
    std::error_code ec;
    if ( !std::filesystem::is_regular_file( zipPath, ec ) )
        return false;
    MR_TIMER;

    struct ZipIcon
    {
        IconTypeData* data = nullptr;
        int sz = 0;
        std::string name;
        std::vector<char> png;
        Expected<Image> image;
    };
    std::vector<ZipIcon> icons;
    auto res = decompressZip( zipPath, [&] ( const std::string& path, std::vector<char>&& content ) -> Expected<void>
    {
        // path is "<icon type folder>/<size folder>/<name>.png"
        const auto parts = split( path, "/" );
        if ( parts.size() != 3 || !isPng( parts[2] ) )
            return {};
        for ( auto& data : data_ )
        {
            if ( utf8string( data.pathDirectory.filename() ) != parts[0] )
                continue;
            for ( int sz = int( data.minMaxSizes.first ); sz <= int( data.minMaxSizes.second ); ++sz )
                if ( parts[1] == sizeSubFolder_( Sizes( sz ) ) )
                    icons.push_back( { &data, sz, utf8string( std::filesystem::path( parts[2] ).stem() ), std::move( content ) } );
            break;
        }
        return {};
    } );
    if ( !res )
    {
        spdlog::error( "Cannot load icons from {}: {}", utf8string( zipPath ), res.error() );
        return false;
    }

    // PNG decoding takes most of the time, so it is parallel; the textures are created on this thread only
    ParallelFor( icons, [&] ( size_t i )
    {
        MemoryStreamBuf buf( icons[i].png );
        std::istream in( &buf );
        icons[i].image = ImageLoad::fromPng( in );
    } );
    int num = 0;
    for ( auto& icon : icons )
    {
        if ( !icon.image )
            continue;
        addIcon_( *icon.data, icon.sz, icon.name, std::move( *icon.image ) );
        ++num;
    }
    spdlog::info( "Loaded {} icons from {}", num, utf8string( zipPath ) );
    for ( auto& data : data_ )
        data.loaded = true;
    return true;
}

void RibbonIcons::load_( IconType type )
{
    size_t num = static_cast< size_t >( type );
    auto& currentData = data_[num];
    if ( currentData.loaded )
        return;
    if ( !zipTried_ )
    {
        zipTried_ = true;
        if ( loadZip_() )
            return;
    }
    currentData.loaded = true;

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
            if ( !entry.is_regular_file( ec ) || !isPng( entry.path() ) )
                continue;
            if ( auto image = ImageLoad::fromPng( entry.path() ) )
                addIcon_( currentData, sz, utf8string( entry.path().stem() ), std::move( *image ) );
        }
    }
}

}
