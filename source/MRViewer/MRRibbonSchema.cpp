#include "MRRibbonSchema.h"
#include "MRRibbonMenuItem.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"
#include "MRViewer/MRViewer.h"
#include "MRRibbonMenu.h"
#include "imgui.h"

namespace MR
{

RibbonSchema& RibbonSchemaHolder::schema()
{
    static RibbonSchema schemaInst;
    return schemaInst;
}

bool RibbonSchemaHolder::addItem( std::shared_ptr<RibbonMenuItem> item )
{
    auto& staticMap = schema().items;
    if ( !item )
        return false;
    if ( staticMap.find( item->name() ) != staticMap.end() )
        return false;

    staticMap[item->name()] = { item };
    return true;
}

void RibbonSchemaLoader::loadSchema() const
{
    auto files = getStructureFiles_( ".items.json" );
    for ( const auto& file : files )
        readItemsJson_( file );

    files = getStructureFiles_( ".ui.json" );
    sortFilesByOrder_( files );
    for ( const auto& file : files )
        readUIJson_( file );
}

void RibbonSchemaLoader::readMenuItemsList( const Json::Value& root, MenuItemsList& resList )
{
    if ( !root.isArray() )
        return;

    MenuItemsList list;

    for ( int i = 0; i <int( root.size() ); ++i )
    {
        auto& item = root[i];
        auto& itemName = item["Name"];
        if ( !itemName.isString() )
        {
            spdlog::warn( "\"Name\" field is not valid or not present in \"Quick Access\" {}", i );
            assert( false );
            continue;
        }
        auto findIt = RibbonSchemaHolder::schema().items.find( itemName.asString() );
        if ( findIt == RibbonSchemaHolder::schema().items.end() )
        {
            spdlog::warn( "Ribbon item \"{}\" is not registered", itemName.asString() );
            // do not assert here because item may have been saved (in user config) before MI version update
            continue;
        }
        list.push_back( itemName.asString() );
    }
    recalcItemSizes();
    resList.insert( resList.end(), std::make_move_iterator( list.begin() ),
                                    std::make_move_iterator( list.end() ) );
}

float sCalcSize( const ImFont* font, const char* begin, const char* end )
{
    float res = font->CalcTextSizeA( font->FontSize, FLT_MAX, -1.0f, begin, end ).x;
    // Round
    // FIXME: This has been here since Dec 2015 (7b0bf230) but down the line we want this out.
    // FIXME: Investigate using ceilf or e.g.
    // - https://git.musl-libc.org/cgit/musl/tree/src/math/ceilf.c
    // - https://embarkstudios.github.io/rust-gpu/api/src/libm/math/ceilf.rs.html
    return float( int( res + +0.99999f ) );
}

SplitCaptionInfo sAutoSplit( const std::string& str, float maxWidth,const ImFont* font, float baseSize )
{
    if ( baseSize < maxWidth )
        return { { str, baseSize } };

    std::vector<std::string_view> substr;

    size_t begin = 0;
    size_t end = str.find( ' ', begin );

    if ( end == std::string::npos )
        return { { str, baseSize } };

    while ( end != std::string::npos )
    {
        if ( begin < end )
            substr.emplace_back( &str[begin], std::distance( &str[begin], &str[end] ) );
        begin = end + 1;
        end = str.find( ' ', begin );
    }
    if ( begin < str.length() - 1 )
        substr.emplace_back( &str[begin], std::distance( str.begin() + begin, str.end() ) );

    std::vector<float> substrWidth;
    for ( const auto& s : substr )
        substrWidth.push_back( sCalcSize( font, &s.front(), &s.back() + 1 ) );

    constexpr const char cSpace[] = " ";
    const float spaceWidth = sCalcSize( font, &cSpace[0], &cSpace[1] );
    size_t index1 = 0;
    size_t index2 = substr.size() - 1;

    SplitCaptionInfo res;
    res.resize( 2 );
    res[0].first = substr[index1];
    res[0].second = substrWidth[index1++];
    res[1].first = substr[index2];
    res[1].second = substrWidth[index2--];
    while ( index1 <= index2 )
    {
        const float predict1 = res[0].second + spaceWidth + substrWidth[index1];
        const float predict2 = res[1].second + spaceWidth + substrWidth[index2];
        if ( predict1 < predict2 )
        {
            const char* start = &res[0].first.front();
            const char* stop = &substr[index1++].back() + 1;
            res[0].first = std::string_view( start, std::distance( start, stop ) );
            res[0].second = predict1;
        }
        else
        {
            const char* start = &substr[index2--].front();
            const char* stop = &res[1].first.back() + 1;
            res[1].first = std::string_view( start, std::distance( start, stop ) );
            res[1].second = predict2;
        }
    }

    return res;
}

void RibbonSchemaLoader::recalcItemSizes()
{
    auto menu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    if ( !menu )
        return;
    const auto& fontManager = menu->getFontManager();

    const float cMaxTextWidth = 
        fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons ) * 
        4 * menu->menu_scaling();

    const auto* font = fontManager.getFontByType( RibbonFontManager::FontType::Small );
    auto& schema = RibbonSchemaHolder::schema();
    for ( auto& item : schema.items )
    {
        if ( !item.second.item )
            continue;

        auto& sizes = item.second.captionSize;

        const auto& caption = item.second.caption.empty() ? item.second.item->name() : item.second.caption;
        sizes.baseSize = sCalcSize( font, caption.data(), caption.data() + caption.size() );
        sizes.splitInfo = sAutoSplit( caption, cMaxTextWidth, font, sizes.baseSize );
    }
}

std::vector<std::filesystem::path> RibbonSchemaLoader::getStructureFiles_( const std::string& fileExtension ) const
{
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for ( auto it = std::filesystem::directory_iterator( GetResourcesDirectory(), ec ); !ec && it != std::filesystem::end( it ); it.increment( ec ) )
    {
        auto filename = it->path().filename().u8string();
        for ( auto& c : filename )
            c = ( char ) tolower( c );
        if ( filename.ends_with( asU8String( fileExtension ) ) )
            files.push_back( it->path() );
    }
    return files;
}

void RibbonSchemaLoader::sortFilesByOrder_( std::vector<std::filesystem::path>& files ) const
{
    std::vector<std::pair<int, int>> order( files.size(), { INT_MAX,0 } );
    for ( int i = 0; i < files.size(); ++i )
    {
        order[i].second = i;
        const auto& file = files[i];
        auto fileJson = deserializeJsonValue( file );
        if ( !fileJson )
        {
            spdlog::error( "JSON ({}) deserialize error: {}", utf8string( file ), fileJson.error() );
            assert( false );
            continue;
        }
        if ( !fileJson.value()["Order"].isInt() )
            continue;
        order[i].first = fileJson.value()["Order"].asInt();
    }
    std::sort( order.begin(), order.end() );
    std::vector<std::filesystem::path> result( files.size() );
    for ( int i = 0; i < result.size(); ++i )
        result[i] = std::move( files[order[i].second] );
    files = std::move( result );
}

void RibbonSchemaLoader::readItemsJson_( const std::filesystem::path& path ) const
{
    auto itemsStructRes = deserializeJsonValue( path );
    if ( !itemsStructRes )
    {
        spdlog::warn( "Cannot parse Json file: {}", utf8string( path ) );
        assert( false );
        return;
    }
    auto items = itemsStructRes.value()["Items"];
    if ( !items.isArray() )
    {
        spdlog::warn( "\"Items\" field is not valid or not present" );
        assert( false );
        return;
    }
    auto itemsSize = int( items.size() );
    for ( int i = 0; i < itemsSize; ++i )
    {
        auto item = items[i];
        auto& itemName = item["Name"];
        auto findIt = RibbonSchemaHolder::schema().items.find( itemName.asString() );
        if ( findIt == RibbonSchemaHolder::schema().items.end() )
        {
#ifndef __EMSCRIPTEN__
            spdlog::warn( "Ribbon item \"{}\" is not registered", itemName.asString() );
            assert( false );
#endif
            continue;
        }
        auto& itemCaption = item["Caption"];
        if ( itemCaption.isString() )
            findIt->second.caption = itemCaption.asString();
        auto itemIcon = item["Icon"];
        if ( !itemIcon.isString() )
        {
            spdlog::warn( "\"Icon\" field is not valid or not present in item: \"{}\"", itemName.asString() );
            assert( false );
        }
        else
            findIt->second.icon = itemIcon.asString();
        auto itemTooltip = item["Tooltip"];
        if ( !itemTooltip.isString() )
        {
            spdlog::warn( "\"Tooltip\" field is not valid or not present in item: \"{}\"", itemName.asString() );
            assert( false );
        }
        else
            findIt->second.tooltip = itemTooltip.asString();
        auto itemDropList = item["DropList"];
        if ( !itemDropList.isArray() )
            continue;
        MenuItemsList dropList;
        auto itemDropListSize = int( itemDropList.size() );
        for ( int j = 0; j < itemDropListSize; ++j )
        {
            auto dropItemName = itemDropList[j]["Name"];
            if ( !dropItemName.isString() )
            {
                spdlog::warn( "\"Name\" field is not valid or not present in drop list of item: \"{}\"", itemName.asString() );
                assert( false );
                continue;
            }
            dropList.push_back( dropItemName.asString() );
        }
        if ( !dropList.empty() && findIt->second.item )
            findIt->second.item->setDropItemsFromItemList( dropList );
    }
}

void RibbonSchemaLoader::readUIJson_( const std::filesystem::path& path ) const
{
    auto itemsStructRes = deserializeJsonValue( path );
    if ( !itemsStructRes )
    {
        spdlog::warn( "Cannot parse Json file: {}", utf8string( path ) );
        assert( false );
        return;
    }
    auto tabs = itemsStructRes.value()["Tabs"];
    if ( !tabs.isArray() )
    {
        spdlog::warn( "\"Tabs\" field is not valid or not present" );
        assert( false );
        return;
    }
    auto tabsSize = int( tabs.size() );
    for ( int i = 0; i < tabsSize; ++i )
    {
        auto tab = tabs[i];
        auto tabName = tab["Name"];
        if ( !tabName.isString() )
        {
            spdlog::warn( "\"Name\" field is not valid or not present in \"Tabs\" {}", i );
            assert( false );
            continue;
        }
        auto groups = tab["Groups"];
        if ( !groups.isArray() )
        {
            spdlog::warn( "\"Groups\" field is not valid or not present in tab: \"{}\"", tabName.asString() );
            assert( false );
            continue;
        }
        auto groupsSize = int( groups.size() );
        if ( groupsSize == 0 )
        {
            spdlog::warn( "\"Groups\" array is empty in tab: \"{}\"", tabName.asString() );
            assert( false );
            continue;
        }
        std::vector<std::string> newGroupsVec;
        for ( int g = 0; g < groupsSize; ++g )
        {
            auto group = groups[g];
            auto groupName = group["Name"];
            if ( !groupName.isString() )
            {
                spdlog::warn( "\"Name\" field is not valid or not present in \"Groups\" {}, in tab: \"{}\"", g, tabName.asString() );
                assert( false );
                continue;
            }
            auto list = group["List"];
            if ( !list.isArray() )
            {
                spdlog::warn( "\"List\" field is not valid or not present in group: \"{}\", in tab: \"{}\"", groupName.asString(), tabName.asString() );
                assert( false );
                continue;
            }
            auto listSize = int( list.size() );
            if ( listSize == 0 )
            {
                spdlog::warn( "\"List\" array is empty in group: \"{}\", in tab: \"{}\"", groupName.asString(), tabName.asString() );
                assert( false );
                continue;
            }
            MenuItemsList items;
            readMenuItemsList( list, items );
            if ( items.empty() )
            {
#ifndef __EMSCRIPTEN__
                spdlog::warn( "\"List\" array has no valid items in group: \"{}\", in tab: \"{}\"", groupName.asString(), tabName.asString() );
                assert( false );
#endif
                continue;
            }
            auto& groupsMapRef = RibbonSchemaHolder::schema().groupsMap[tabName.asString() + groupName.asString()];
            if ( groupsMapRef.empty() )
            {
                groupsMapRef = std::move( items );
                newGroupsVec.push_back( groupName.asString() );
            }
            else
            {
                groupsMapRef.insert( groupsMapRef.end(), items.begin(), items.end() );
            }
        }
        if ( newGroupsVec.empty() )
            continue; // it can be ok for additional ui.json files
        auto& tabRef = RibbonSchemaHolder::schema().tabsMap[tabName.asString()];
        if ( tabRef.empty() )
        {
            RibbonSchemaHolder::schema().tabsOrder.push_back( tabName.asString() );
            tabRef = std::move( newGroupsVec );
        }
        else
        {
            tabRef.insert( tabRef.end(), newGroupsVec.begin(), newGroupsVec.end() );
        }
    }

    if ( itemsStructRes.value().isMember( "Quick Access" ) )
        readMenuItemsList( itemsStructRes.value()["Quick Access"], RibbonSchemaHolder::schema().defaultQuickAccessList );

    if ( itemsStructRes.value().isMember( "Header Quick Access" ) )
        readMenuItemsList( itemsStructRes.value()["Header Quick Access"], RibbonSchemaHolder::schema().headerQuickAccessList );

    if ( itemsStructRes.value().isMember( "Scene Buttons" ) )
        readMenuItemsList( itemsStructRes.value()["Scene Buttons"], RibbonSchemaHolder::schema().sceneButtonsList );
}

}
