#pragma once

#include "exports.h"
#include "MRRibbonMenuItem.h"

namespace MR
{

class RibbonMenuItemAdder
{
public:
    /// calls RibbonSchemaHolder::addItem( item_ = item );
    MRVIEWER_API RibbonMenuItemAdder( std::shared_ptr<RibbonMenuItem> item );

    /// calls RibbonSchemaHolder::delItem( item_ );
    MRVIEWER_API ~RibbonMenuItemAdder();

private:
    std::shared_ptr<RibbonMenuItem> item_;
};

template<typename T>
class RibbonMenuItemAdderT : RibbonMenuItemAdder
{
public:
    static_assert( std::is_base_of_v<RibbonMenuItem, T> );

    template<typename... Args>
    RibbonMenuItemAdderT( Args&&... args ) : RibbonMenuItemAdder( std::make_shared<T>( std::forward<Args>( args )... ) )
    {
    }
};

/// registers plugin on module loading, and unregister plugin on module unloading
#define MR_REGISTER_RIBBON_ITEM(pluginType) \
    static RibbonMenuItemAdderT<pluginType> ribbonMenuItemAdder##pluginType##_;

} //namespace MR
