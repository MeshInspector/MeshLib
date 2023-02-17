#pragma once
#include "MRSceneStateCheck.h"
#include "MRViewerEventsListener.h"
#include <string>
#include <memory>

namespace MR
{

using MenuItemsList = std::vector<std::string>;

class Object;

enum class RibbonItemType
{
    Button,
    ButtonWithDrop
};

// class to hold menu items
// some information stored in json (icons path, tab name, subtab name)
class RibbonMenuItem : virtual public ISceneStateCheck
{
public:
    MR_DELETE_MOVE( RibbonMenuItem );

    RibbonMenuItem( std::string name ) :
        name_{ std::move( name ) }
    {}

    virtual ~RibbonMenuItem() = default;
    
    // returns true if state of item changed
    virtual bool action() = 0;
    // for state items returns true if activated
    virtual bool isActive() const { return false; }
    // true if this item is blocking (only one blocking item can be active at once)
    virtual bool blocking() const { return false; }

    const std::string& name() const { return name_; }

    void setRibbonItemType( RibbonItemType type ) { type_ = type; }

    // type of this item, base RibbonMenuItem can be only button
    virtual RibbonItemType type() const { return type_; }

    using DropItemsList = std::vector<std::shared_ptr<RibbonMenuItem>>;

    // set drop list by found in RibbonSchema items
    // !note also set type_ to ButtonWithDrop
    MRVIEWER_API void setDropItemsFromItemList( const MenuItemsList& itemsList );

    // returns list of stored ribbon items to drop
    // !note that this function can be called each frame for opened drop list
    virtual const DropItemsList& dropItems() const { return dropList_; };

    // return not-empty string with tooltip that shall replace the static tooltip from json
    virtual std::string getDynamicTooltip() const { return {}; }

protected:
    RibbonItemType type_{ RibbonItemType::Button };
    DropItemsList dropList_;
private:
    std::string name_; // key to find in holder and json
};

// Simple ribbon item acting given lambda
class LambdaRibbonItem : public RibbonMenuItem
{
public:
    using SimpleLambda = std::function<void()>;
    LambdaRibbonItem( std::string name, SimpleLambda lambda ) :
        RibbonMenuItem( std::move( name ) ),
        lambda_( std::move( lambda ) )
    {}

    virtual ~LambdaRibbonItem() = default;

    virtual bool action() override
    {
        lambda_();
        return false;
    }
private:
    SimpleLambda lambda_;
};

}