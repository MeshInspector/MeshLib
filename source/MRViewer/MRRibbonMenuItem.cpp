#include "MRRibbonMenuItem.h"
#include "MRRibbonSchema.h"
#include "MRRibbonMenu.h"
#include <cassert>

namespace MR
{


RibbonMenuItem::RibbonMenuItem( std::string name ) :
    name_{ std::move( name ) }
{
}

void RibbonMenuItem::registerShortcut( RibbonMenu& menu, const ShortcutConfig& conf )
{
    if ( auto shortcut = defaultShortcut_( conf ) )
        menu.addRibbonItemShortcut( name_, *shortcut );
}

std::optional<Shortcut> RibbonMenuItem::defaultShortcut_( const ShortcutConfig& ) const
{
    return {}; // most items have no default shortcut
}

void RibbonMenuItem::setDropItemsFromItemList( const MenuItemsList& itemsList )
{
    dropList_.clear();
    const auto& schema = RibbonSchemaHolder::schema();
    for ( const auto& itemName : itemsList )
    {
        auto itemIt = schema.items.find( itemName );
        if ( itemIt == schema.items.end() )
            continue;
        if ( !itemIt->second.item )
            continue;
        dropList_.push_back( itemIt->second.item );
    }
    if ( !dropList_.empty() )
        type_ = RibbonItemType::ButtonWithDrop;
}

}
