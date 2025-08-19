#include "MRRibbonSchema.h"
#include "MRLambdaRibbonItem.h"
#include "MRImGui.h"
#include "MRRibbonMenu.h"
#include "MRViewer.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"
#include "MRSceneCache.h"

namespace MR
{

void RibbonSchema::eliminateEmptyGroups()
{
    // eliminate empty groups
    for ( auto it = groupsMap.begin(); it != groupsMap.end(); )
    {
        if ( it->second.empty() )
        {
            spdlog::info( "Empty group {} eliminated", it->first );
            it = groupsMap.erase( it );
        }
        else
            ++it;
    }

    // eliminate references on not-existing groups
    for ( auto & [tabName, groups] : tabsMap )
    {
        std::erase_if( groups, [this, &tabName = tabName]( const std::string & groupName ) { return !groupsMap.contains( tabName + groupName ); } );
    }
}

void RibbonSchema::sortTabsByPriority()
{
    std::stable_sort( tabsOrder.begin(), tabsOrder.end(), [] ( const auto& a, const auto& b )
    {
        return a.priority < b.priority;
    } );
}

void RibbonSchema::updateCaptions()
{
    for ( const auto& [name, item] : items )
    {
        auto statePlugin = std::dynamic_pointer_cast< StateBasePlugin >( item.item );
        if ( !statePlugin )
            continue;
        statePlugin->setUIName( item.caption.empty() ? name : item.caption );
    }
}

RibbonSchema& RibbonSchemaHolder::schema()
{
    static RibbonSchema schemaInst;
    return schemaInst;
}

bool RibbonSchemaHolder::addItem( const std::shared_ptr<RibbonMenuItem>& item )
{
    auto& staticMap = schema().items;
    if ( !item )
        return false;

    auto [it, inserted] = staticMap.insert( { item->name(), MenuItemInfo{ item } } );
    if ( !inserted )
    {
        spdlog::warn( "Attempt to register again ribbon item {}", item->name() );
        return false;
    }

#ifndef NDEBUG
    spdlog::info( "Register ribbon item {}", item->name() );
#endif
    return true;
}

bool RibbonSchemaHolder::delItem( const std::shared_ptr<RibbonMenuItem>& item )
{
    auto& staticMap = schema().items;
    if ( !item )
        return false;

    const auto it = staticMap.find( item->name() );
    if ( it == staticMap.end() || it->second.item != item )
    {
        spdlog::warn( "Attempt to unregister missing ribbon item {}", item->name() );
        return false;
    }

    staticMap.erase( it );
#ifndef NDEBUG
    spdlog::info( "Unregister ribbon item {}, use count={}", item->name(), item.use_count() );
#endif
    return true;
}

std::vector<RibbonSchemaHolder::SearchResult> RibbonSchemaHolder::search( const std::string& searchStr, const SearchParams& params )
{
    std::vector<std::pair<SearchResult, SearchResultWeight>> rawResult;
    
    if ( searchStr.empty() )
        return {};
    auto words = split( searchStr, " " );
    std::erase_if( words, [] ( const auto& str ) { return str.empty(); } );
    auto calcWeight = [&words] ( const std::string& sourceStr )->Vector2f
    {
        if ( sourceStr.empty() )
            return { 1.0f, 1.f };
        
        auto sourceWords = split( sourceStr, " " );
        std::erase_if( sourceWords, [] ( const auto& str ) { return str.empty(); } );
        if ( sourceWords.empty() )
            return { 1.0f, 1.f };

        const int sourceWordsSize = int( sourceWords.size() );
        //std::vector<int> errorArr( words.size() * sourceWordsSize, -1 );
        std::vector<bool> busyWord( sourceWordsSize, false );
        int sumError = 0;
        int searchCharCount = 0;
        int posWeight = sourceWordsSize;
        for ( int i = 0; i < words.size(); ++i )
        {
            searchCharCount += int( words[i].size() );
            int minError = int( words[i].size() );
            int minErrorIndex = -1;
            for ( int j = 0; j < sourceWordsSize; ++j )
            {
                if ( busyWord[j] )
                    continue;
                int cornersInsertions = 0;
                int error = calcDamerauLevenshteinDistance( words[i], sourceWords[j], false, &cornersInsertions );
                if ( i == words.size() - 1 )
                    error -= cornersInsertions;
                if ( error < minError )
                {
                    minError = error;
                    minErrorIndex = j;
                }
            }
            if ( minErrorIndex != -1 )
            {
                busyWord[minErrorIndex] = true;
                posWeight += minErrorIndex;
            }
            sumError += minError;
        }
        return { std::clamp( float( sumError ) / searchCharCount, 0.0f, 1.0f ), float( posWeight ) / sourceWordsSize / words.size()};
    };

    const float maxWeight = 0.25f;
    bool exactMatch = false;
    // check item (calc difference from search item) and add item to raw results if difference less than threshold
    auto checkItem = [&] ( const MenuItemInfo& item, int t )
    {
        const auto& caption = item.caption.empty() ? item.item->name() : item.caption;
        const auto& tooltip = item.tooltip;
        std::pair<SearchResult, SearchResultWeight> itemRes;
        itemRes.first.tabIndex = t;
        itemRes.first.item = &item;
        const auto posCE = findSubstringCaseInsensitive( caption, searchStr );
        if ( posCE != std::string::npos )
        {
            if ( !exactMatch )
            {
                rawResult.clear();
                exactMatch = true;
            }
            itemRes.second.captionWeight = 0.f;
            itemRes.second.captionOrderWeight = float( posCE ) / caption.size();
            rawResult.push_back( itemRes );
            return;
        }
        else if ( exactMatch )
        {
            const auto posTE = findSubstringCaseInsensitive( tooltip, searchStr );
            if ( posTE == std::string::npos )
                return;
            itemRes.second.tooltipWeight = 0.f;
            itemRes.second.tooltipOrderWeight = float( posTE ) / tooltip.size();
            rawResult.push_back( itemRes );
            return;
        }

        Vector2f weightEP = calcWeight( caption );
        itemRes.second.captionWeight = weightEP.x;
        itemRes.second.captionOrderWeight = weightEP.y;
        weightEP = calcWeight( tooltip );
        itemRes.second.tooltipWeight = weightEP.x;
        itemRes.second.tooltipOrderWeight = weightEP.y;
        if ( itemRes.second.captionWeight > maxWeight && itemRes.second.tooltipWeight > maxWeight )
            return;
        rawResult.push_back( itemRes );
    };
    const auto& schema = RibbonSchemaHolder::schema();
    auto lookUpMenuItemList = [&] ( const MenuItemsList& list, int t )
    {
        for ( int i = 0; i < list.size(); ++i )
        {
            auto item = schema.items.find( list[i] );
            if ( item == schema.items.end() || !item->second.item )
                continue;
            checkItem( item->second, t );
            if ( item->second.item->type() == RibbonItemType::ButtonWithDrop )
            {
                for ( const auto& dropRibItem : item->second.item->dropItems() )
                {
                    if ( !dropRibItem )
                        continue;
                    if ( std::dynamic_pointer_cast< LambdaRibbonItem >( dropRibItem ) )
                        continue;
                    auto dropItem = schema.items.find( dropRibItem->name() );
                    if ( dropItem == schema.items.end() || !dropItem->second.item )
                        continue;
                    checkItem( dropItem->second, t );
                }
            }
        }
    };
    for ( int t = 0; t < schema.tabsOrder.size(); ++t )
    {
        if ( schema.tabsOrder[t].experimental && !getViewerInstance().experimentalFeatures )
            continue;
        auto tabItem = schema.tabsMap.find( schema.tabsOrder[t].name );
        if ( tabItem == schema.tabsMap.end() )
            continue;
        for ( int g = 0; g < tabItem->second.size(); ++g )
        {
            auto groupItem = schema.groupsMap.find( schema.tabsOrder[t].name + tabItem->second[g] );
            if ( groupItem == schema.groupsMap.end() )
                continue;
            lookUpMenuItemList( groupItem->second, t );
        }
    }
    lookUpMenuItemList( schema.headerQuickAccessList, -1 );
    lookUpMenuItemList( schema.sceneButtonsList, -1 );

    // clear duplicated results
    std::sort( rawResult.begin(), rawResult.end(), [] ( const auto& a, const auto& b )
    {
        // tab order sorting has been added to stabilize results for similar queries (i.e. "c" / "cl" / "clo" / "clone", i6438 )
        const auto ptrA = intptr_t( a.first.item );
        const auto ptrB = intptr_t( b.first.item );
        const auto& tabIndexA = a.first.tabIndex;
        const auto& tabIndexB = b.first.tabIndex;
        return ptrA < ptrB || ( ptrA == ptrB && tabIndexA < tabIndexB );
    } );
    rawResult.erase(
        std::unique( rawResult.begin(), rawResult.end(),
            [] ( const auto& a, const auto& b )
    {
        return a.first.item == b.first.item;
    } ),
        rawResult.end() );

    std::sort( rawResult.begin(), rawResult.end(), [maxWeight, requirementsFunc = params.requirementsFunc] ( const auto& a, const auto& b )
    {
        const bool aCaptionWeightCorrect = a.second.captionWeight <= maxWeight;
        const bool bCaptionWeightCorrect = b.second.captionWeight <= maxWeight;

        // 1 sort priority
        // the corresponding caption takes precedence over the corresponding tooltip
        if ( aCaptionWeightCorrect != bCaptionWeightCorrect )
            return aCaptionWeightCorrect;

        if ( requirementsFunc )
        {
            const bool aAvailable = requirementsFunc( a.first.item->item ).empty();
            const bool bAvailable = requirementsFunc( b.first.item->item ).empty();

            // 2 sort priority
            // available tool takes precedence over unavailable
            if ( aAvailable != bAvailable )
                return aAvailable;
        }

        // 3 sort priority
        // if both have the correct caption weight, then compare by caption, otherwise compare by tooltip
        const auto& aWeight = aCaptionWeightCorrect ? a.second.captionWeight : a.second.tooltipWeight;
        const auto& bWeight = aCaptionWeightCorrect ? b.second.captionWeight : b.second.tooltipWeight;
        // 4 sort priority
        // if both have the same weight, then compare by order weight
        const auto& aOrderWeight = aCaptionWeightCorrect ? a.second.captionOrderWeight : a.second.tooltipOrderWeight;
        const auto& bOrderWeight = aCaptionWeightCorrect ? b.second.captionOrderWeight : b.second.tooltipOrderWeight;
        return std::tuple( aWeight, aOrderWeight ) < std::tuple( bWeight, bOrderWeight );
    } );

    // filter results with error threshold as 3x minimum caption error 
    if ( !rawResult.empty() && rawResult[0].second.captionWeight < maxWeight / 3.f )
    {
        const float maxWeightNew = rawResult[0].second.captionWeight * 3.f;
        if ( rawResult.back().second.captionWeight > maxWeightNew )
        {
            auto tailIt = std::find_if( rawResult.begin(), rawResult.end(), [&] ( const auto& a )
            {
                return a.second.captionWeight > maxWeightNew && a.second.tooltipWeight > maxWeightNew;
            } );
            rawResult.erase( tailIt, rawResult.end() );
        }
    }

    std::vector<SearchResult> res( rawResult.size() );
    if ( params.weights )
        *params.weights = std::vector<SearchResultWeight>( rawResult.size() );
    if ( params.captionCount )
        *params.captionCount = -1;
    for ( int i = 0; i < rawResult.size(); ++i )
    {
        if ( !rawResult[i].first.item )
        {
            assert( false );
            continue;
        }
        res[i] = rawResult[i].first;
        if ( params.captionCount && rawResult[i].second.captionWeight > maxWeight &&
            ( i == 0 || ( i > 0 && rawResult[i-1].second.captionWeight <= maxWeight ) ) )
            *params.captionCount = i;
        if ( params.weights )
            ( *params.weights )[i] = rawResult[i].second;
    }

    return res;
}

int RibbonSchemaHolder::findItemTab( const std::shared_ptr<RibbonMenuItem>& item )
{
    if ( !item )
        return -1;

    const auto& schema = RibbonSchemaHolder::schema();
    for ( int t = 0; t < schema.tabsOrder.size(); ++t )
    {
        if ( schema.tabsOrder[t].experimental && !getViewerInstance().experimentalFeatures )
            continue;
        auto gpIt = schema.tabsMap.find( schema.tabsOrder[t].name );
        if ( gpIt == schema.tabsMap.end() )
            continue;
        for ( const auto& gp : gpIt->second )
        {
            auto itmesIt = schema.groupsMap.find( schema.tabsOrder[t].name + gp );
            if ( itmesIt == schema.groupsMap.end() )
                continue;
            for ( const auto& itemName : itmesIt->second )
            {
                if ( item->name() == itemName )
                    return t;
            }
        }
    }
    return -1;
}

void RibbonSchemaLoader::loadSchema() const
{
    MR_TIMER;
    auto files = getStructureFiles_( ".items.json" );
    if ( files.empty() )
        spdlog::error( "No Ribbon Items files found" );
    for ( const auto& file : files )
    {
        spdlog::info( "Reading {}", utf8string( file ) );
        readItemsJson_( file );
    }

    files = getStructureFiles_( ".ui.json" );
    if ( files.empty() )
        spdlog::error( "No Ribbon UI files found" );
    sortFilesByOrder_( files );
    for ( const auto& file : files )
    {
        spdlog::info( "Reading {}", utf8string( file ) );
        readUIJson_( file );
    }
    spdlog::info( "Reading Ribbon Schema done" );

    RibbonSchemaHolder::schema().eliminateEmptyGroups();
    RibbonSchemaHolder::schema().sortTabsByPriority();
    RibbonSchemaHolder::schema().updateCaptions();
}

void RibbonSchemaLoader::readMenuItemsList( const Json::Value& root, MenuItemsList& list )
{
    if ( !root.isArray() )
        return;

    list.clear();

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
    auto menu = getViewerInstance().getMenuPlugin();
    if ( !menu )
        return;
    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small );
    if ( !font )
        return;

    const float cMaxTextWidth = 
        RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons ) * 
        4 * menu->menu_scaling();

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
    for ( auto entry : Directory{ SystemPath::getResourcesDirectory(), ec } )
    {
        auto filename = entry.path().filename().u8string();
        for ( auto& c : filename )
            c = ( char ) tolower( c );
        if ( filename.ends_with( asU8String( fileExtension ) ) )
            files.push_back( entry.path() );
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
    readItemsJson_( *itemsStructRes );
}

void RibbonSchemaLoader::readItemsJson_( const Json::Value& itemsStruct ) const
{
    auto items = itemsStruct["Items"];
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
        auto& [_, menuItem] = *findIt;

        auto& itemCaption = item["Caption"];
        if ( itemCaption.isString() )
            menuItem.caption = itemCaption.asString();

        auto& itemHelpLink = item["HelpLink"];
        if ( itemHelpLink.isString() )
            menuItem.helpLink = itemHelpLink.asString();

        auto itemIcon = item["Icon"];
        if ( !itemIcon.isString() )
        {
            spdlog::warn( "\"Icon\" field is not valid or not present in item: \"{}\"", itemName.asString() );
            assert( false );
        }
        else
            menuItem.icon = itemIcon.asString();

        auto itemTooltip = item["Tooltip"];
        if ( !itemTooltip.isString() )
        {
            spdlog::warn( "\"Tooltip\" field is not valid or not present in item: \"{}\"", itemName.asString() );
            assert( false );
        }
        else
            menuItem.tooltip = itemTooltip.asString();

        auto itemDropList = item["DropList"];
        if ( itemDropList.isArray() && menuItem.item )
        {
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
            if ( !dropList.empty() )
                menuItem.item->setDropItemsFromItemList( dropList );
        }

        if ( auto loadListener = std::dynamic_pointer_cast<RibbonSchemaLoadListener>( menuItem.item ) )
            loadListener->onRibbonSchemaLoad_();
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
    readUIJson_( *itemsStructRes );
}

void RibbonSchemaLoader::readUIJson_( const Json::Value& itemsStructure ) const
{
    auto tabs = itemsStructure["Tabs"];
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
        auto tabPriorityJSON = tab["Priority"];
        bool experementalTab = false;
        if ( tab["Experimental"].isBool() )
            experementalTab = tab["Experimental"].asBool();
        int tabPriority{ 0 };
        if ( tabPriorityJSON.isInt() )
            tabPriority = tabPriorityJSON.asInt();
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
            MenuItemsList items;
            readMenuItemsList( list, items );
            auto [it, inserted] = RibbonSchemaHolder::schema().groupsMap.insert( { tabName.asString() + groupName.asString(), MenuItemsList{} } );
            auto& groupsMapRef = it->second;
            if ( inserted )
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
            RibbonSchemaHolder::schema().tabsOrder.push_back( { tabName.asString(),tabPriority,experementalTab } );
            tabRef = std::move( newGroupsVec );
        }
        else
        {
            auto it = std::find_if( 
                RibbonSchemaHolder::schema().tabsOrder.begin(), 
                RibbonSchemaHolder::schema().tabsOrder.end(),
                [&] ( const RibbonTab& tnp ) { return tnp.name == tabName.asString(); } );

            if ( it != RibbonSchemaHolder::schema().tabsOrder.end() && tabPriority != 0 )
                it->priority = tabPriority;

            tabRef.insert( tabRef.end(), newGroupsVec.begin(), newGroupsVec.end() );
        }
    }

    auto loadQuickAccess = [&] ( const std::string& key, MenuItemsList& oldList )
    {
        if ( itemsStructure.isMember( key ) )
        {
            MenuItemsList newDefaultList;
            readMenuItemsList( itemsStructure[key], newDefaultList );
            // move items of `newDefaultList` that are not preset in `oldList` to the end of `oldList`
            std::copy_if(
                std::make_move_iterator( newDefaultList.begin() ),
                std::make_move_iterator( newDefaultList.end() ),
                std::back_inserter( oldList ),
                [&] ( const auto& newDefItem )
            {
                return std::none_of(
                    oldList.begin(),
                    oldList.end(),
                    [&] ( const auto& oldDefItem )
                {
                    return newDefItem == oldDefItem;
                }
                );
            } );
        }
    };

    loadQuickAccess( "Quick Access", RibbonSchemaHolder::schema().defaultQuickAccessList );
    loadQuickAccess( "Header Quick Access", RibbonSchemaHolder::schema().headerQuickAccessList );
    loadQuickAccess( "Scene Buttons", RibbonSchemaHolder::schema().sceneButtonsList );
}

}
