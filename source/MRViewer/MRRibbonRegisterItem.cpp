#include "MRRibbonRegisterItem.h"
#include "MRRibbonSchema.h"

namespace MR
{

RibbonMenuItemAdder::RibbonMenuItemAdder( std::shared_ptr<RibbonMenuItem> item ) : item_( std::move( item ) )
{
    RibbonSchemaHolder::addItem( item_ );
}

RibbonMenuItemAdder::~RibbonMenuItemAdder()
{
    RibbonSchemaHolder::delItem( item_ );
}

} //namespace MR
