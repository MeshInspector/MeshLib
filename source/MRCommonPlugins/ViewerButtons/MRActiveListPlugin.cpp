#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewer.h"

namespace MR
{

class ActivePluginsList : public RibbonMenuItem
{
public:
    ActivePluginsList() :
        RibbonMenuItem( "Active Plugins List" )
    {
    }
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override
    {
        auto menu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
        if ( !menu )
            return "No menu present";
        if ( !menu->hasAnyActiveItem() )
            return "No active tools.";
        return "";
    }

    virtual bool action() override
    {
        auto menu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
        if ( !menu )
            return false;
        menu->showActiveList();
        return false;
    }
};

MR_REGISTER_RIBBON_ITEM( ActivePluginsList )

}