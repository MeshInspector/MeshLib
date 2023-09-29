#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRCommandLoop.h"

namespace MR
{

class ActivePluginsList : public RibbonMenuItem
{
public:
    ActivePluginsList() :
        RibbonMenuItem( "Active Plugins List" )
    {
        CommandLoop::appendCommand( [&]
        {
            menu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>();
        } );
    }
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override
    {
        if ( !menu_ )
            return "No menu present";
        if ( !menu_->hasAnyActiveItem() )
            return "No active tools.";
        return "";
    }

    virtual bool action() override
    {
        if ( !menu_ )
            return false;
        menu_->showActiveList();
        return false;
    }
private:
    std::shared_ptr<RibbonMenu> menu_{ nullptr };
};

MR_REGISTER_RIBBON_ITEM( ActivePluginsList )

}