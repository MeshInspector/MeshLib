#include "MRRibbonMenuItem.h"
#include "MRRibbonSchema.h"

namespace MR
{

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