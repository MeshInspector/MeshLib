#include "MRStatePlugin.h"
#include "MRMesh/MRString.h"
#include "MRRibbonMenu.h"
#include "MRMesh/MRSystem.h"
#include "MRCommandLoop.h"
#include "MRMesh/MRConfig.h"
#include "imgui.h"
#include "imgui_internal.h"

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
    CommandLoop::appendCommand( [this] ()
    {
        std::string name = this->name();
        auto item = RibbonSchemaHolder::schema().items.find( name );
        if ( item != RibbonSchemaHolder::schema().items.end() )
        {
            if ( !item->second.caption.empty() )
                name = item->second.caption;
        }
        plugin_name = std::move( name );
        plugin_name += UINameSuffix();
    }, CommandLoop::StartPosition::AfterPluginInit );
    tab_ = tab;
}

void StateBasePlugin::drawDialog( ImGuiContext* )
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
        else if ( !dialogIsOpen_ )
        {
            dialogIsOpen_ = true; // we are here after setting `dialogIsOpen_ = false` followed by `onDisable_() = false`
        }
    }
    if ( res )
    {
        if ( auto ribbonMenu = RibbonMenu::instance() )
            ribbonMenu->updateItemStatus( name() );
    }
    return res;
}

bool StateBasePlugin::dialogIsOpen() const
{
    return dialogIsOpen_ && !shouldClose_(); // virtual call from IPluginCloseCheck
}

const char* StateBasePlugin::UINameSuffix()
{
    return "##CustomStatePlugin";
}

void StateBasePlugin::setUIName( std::string name )
{
    plugin_name = std::move( name );
    plugin_name += UINameSuffix();
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
    return ( findSubstringCaseInsensitive( name(), mask) != std::string::npos ) ||
        ( findSubstringCaseInsensitive( getTooltip(), mask ) != std::string::npos );
}

bool StateBasePlugin::ImGuiBeginWindow_( ImGui::CustomStatePluginWindowParameters params )
{
    if ( !params.collapsed )
        params.collapsed = &dialogIsCollapsed_;

    if ( !params.helpBtnFn )
    {
        auto it = RibbonSchemaHolder::schema().items.find( name() );
        if ( it != RibbonSchemaHolder::schema().items.end() && !it->second.helpLink.empty() )
            params.helpBtnFn = [&] () { OpenLink( it->second.helpLink ); };
    }

    return BeginCustomStatePlugin( uiName().c_str(), &dialogIsOpen_, params );
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
