#pragma once

#include "MRViewerPlugin.h"
#include "MRSceneStateCheck.h"
#include "MRStatePluginAutoClose.h"
#include "MRViewerEventsListener.h"
#include "MRRibbonMenuItem.h"
#include <filesystem>

struct ImGuiContext;

namespace MR
{
/*
Inheritance diagram:
   
   EP        - End Plugin             
   SLP       - StateListenerPlugin (optional), if not present EP->SBP
   ML        - MultiListener
   Con       - Connectables (can be a lot of them)
   ConHold   - ConnectionHolder (one for each Connectable)
   ICon      - IConnectable virtual (one for all ConnectionHolder)             
   SBP       - StateBasePlugin (pointers of this type stored in Menu)
   VP        - ViewerPlugin
   RMI       - RibbonMenuItem
   ISSC      - ISceneStateCheck virtual
   SSC Block - Block of SceneStateCheck (this block can have different topology)
               implements `isAvailable` function from ISSC, that is called from Menu's SBP* 
               (leads to: warning C4250: 'EP': inherits 'SSC' via dominance, OK on clang and gcc)
   IPCC      - IPluginCloseCheck virtual
   PCC       - Block of PluginCloseCheck (this block can have different topology)
               implements `shouldClose_` `onPluginEnable_` and `onPluginDisable_`
               functions from IPCC, that is called from Menu's SBP*
               (leads to: warning C4250: 'EP': inherits 'PCC' via dominance, OK on clang and gcc)

       . ICon
      /|\(virtual)   ___________. ISSC
     / | \          /           |(virt)
    .  .  . ConHold/            |
    |  |  |       /             |
    |  |  |  VP  /              |
Con .  .  .  .  .RMI            |
     \ | /   | /                |
      \|/    |/   IPCC          | 
    ML . SBP .--. (virt)     __/
        \   /  /          __/
         \ /  /        __/
      SLP .  . PCC  __. SSC Block
           \ |   __/
            \|__/
             . EP
*/


enum class StatePluginTabs
{
    Basic,
    Mesh,
    DistanceMap,
    PointCloud,
    Selection,
    Voxels,
    Analysis,
    Test,
    Other,
    Count
};

class Object;

class MRVIEWER_CLASS StateBasePlugin :public ViewerPlugin, public RibbonMenuItem, public virtual IPluginCloseCheck
{
public:
    MRVIEWER_API StateBasePlugin( std::string name, StatePluginTabs tab = StatePluginTabs::Other );
    virtual ~StateBasePlugin() = default;

    MRVIEWER_API virtual void drawDialog( float menuScaling, ImGuiContext* ctx );

    virtual bool action() override { return enable( !isEnabled() ); }

    virtual bool isActive() const override { return isEnabled(); };

    virtual bool blocking() const override { return true; }

    MRVIEWER_API virtual void shutdown() override;

    MRVIEWER_API bool isEnabled() const;
    MRVIEWER_API virtual bool enable( bool on );

    MRVIEWER_API virtual bool dialogIsOpen() const;

    MRVIEWER_API StatePluginTabs getTab() const;

    MRVIEWER_API static const char* getTabName( StatePluginTabs tab );

    MRVIEWER_API virtual std::string getTooltip() const;

    // returns special string for sorting plugins in menu (plugin name by default)
    virtual std::string sortString() const { return name(); }

    // check if search mask satisfies for this plugin
    MRVIEWER_API bool checkStringMask( const std::string& mask ) const;

protected:
    MRVIEWER_API virtual bool onEnable_();
    MRVIEWER_API virtual bool onDisable_();

    bool isEnabled_{false};
    bool dialogIsOpen_{false};
    bool dialogIsCollapsed_{ false };

    StatePluginTabs tab_{StatePluginTabs::Other};
};

template<typename ...Connectables>
class StateListenerPlugin : public StateBasePlugin, public MultiListener<Connectables...>
{
public:
    using StateBasePlugin::StateBasePlugin;
    virtual ~StateListenerPlugin() = default;

    using PluginParent = StateListenerPlugin<Connectables...>;
    using MultiListenerBase = MultiListener<Connectables...>;

    virtual bool enable( bool on ) override final
    {
        if ( StateBasePlugin::enable( on ) )
        {
            if ( on )
                // 10th group to have some extra space before it (at_front, each new active plugin to be first in listeners list)
                MultiListenerBase::connect( viewer, 10, boost::signals2::at_front );
            else
                MultiListenerBase::disconnect();
            return true;
        }
        return false;
    }
};

}
