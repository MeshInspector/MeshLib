#include "MRStatePlugin.h"
#include "MRMesh/MRString.h"
#include "MRRibbonMenu.h"
#include "MRViewer.h"

namespace MR
{
constexpr std::array<const char*, size_t( StatePluginTabs::Count )> TabsNames =
{
    "Basic",
    "Mesh",
    "DistanceMap",
    "PointCloud",
    "Selection",
    "Voxels",
    "Analysis",
    "Test",
    "Other"
};

StateBasePlugin::StateBasePlugin( std::string name, StatePluginTabs tab ):
    ViewerPlugin(),
    RibbonMenuItem( name )
{
    plugin_name = std::move( name );
    tab_ = tab;
}

void StateBasePlugin::drawDialog( float, ImGuiContext* )
{
}

bool StateBasePlugin::isEnabled() const
{
    return isEnabled_;
}

bool StateBasePlugin::enable( bool on )
{
    bool res = false;
    if ( on && !isEnabled_ )
    {
        if ( onEnable_() )
        {
            isEnabled_ = true;
            dialogIsOpen_ = true;
            onPluginEnable_(); // virtual call from IPluginCloseCheck
            res = true;
        }
    }
    else if ( !on && isEnabled_ )
    {
        if ( onDisable_() )
        {
            isEnabled_ = false;
            dialogIsOpen_ = false;
            onPluginDisable_(); // virtual call from IPluginCloseCheck
            res = true;
        }
    }
    if ( res )
    {
        auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
        if ( ribbonMenu )
            ribbonMenu->updateItemStatus( plugin_name );
    }
    return res;
}

bool StateBasePlugin::dialogIsOpen() const
{
    return dialogIsOpen_ && !shouldClose_(); // virtual call from IPluginCloseCheck
}

StatePluginTabs  StateBasePlugin::getTab() const
{
    return tab_;
}

const char* StateBasePlugin::getTabName( StatePluginTabs tab )
{
    return TabsNames[int( tab )];
}

void StateBasePlugin::shutdown()
{
    if ( isEnabled_ )
        enable( false );
}

bool StateBasePlugin::checkStringMask( const std::string& mask ) const
{
    return ( findSubstringCaseInsensitive( plugin_name, mask ) != std::string::npos ) ||
        ( findSubstringCaseInsensitive( getTooltip(), mask ) != std::string::npos );
}

std::string StateBasePlugin::getTooltip() const
{
    return {};
}

bool StateBasePlugin::onEnable_()
{
    return true;
}

bool StateBasePlugin::onDisable_()
{
    return true;
}

}