#include "MRViewer/MRViewerPlugin.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRMouseController.h"

namespace MR
{

// Plugin that checks mouse usage conflicts between camera operations and current plugin
class MouseControlChecker : public ViewerPlugin
{
public:
    MouseControlChecker()
    {}
    virtual void init( Viewer* _viewer ) override;
    virtual void shutdown() override;

private:
    boost::signals2::connection connection_;
    size_t mouseDownConnections_{};

    bool checkUseLeftButton_();
    void pluginEnabledChanged_( bool enabled );
};

void MouseControlChecker::init( Viewer* _viewer )
{
    viewer = _viewer;
    RibbonMenu* menu = getViewerInstance().getMenuPluginAs<RibbonMenu>().get();
    if ( menu )
        connection_ = menu->pluginEnabledChangedSignal.connect( 0, [this] ( StateBasePlugin*, bool enabled )
        {
            pluginEnabledChanged_( enabled );
        } );

    mouseDownConnections_ = viewer->mouseDownSignal.num_slots();
}
void MouseControlChecker::shutdown()
{
    connection_.disconnect();
}

bool MouseControlChecker::checkUseLeftButton_()
{
    // Check if camera movement is set to use left mouse button, regardless of modifiers
    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        if ( mode == MouseMode::None )
            continue;
        auto ctrl = viewer->mouseController().findControlByMode( mode );
        if ( ctrl && ctrl->btn == MouseButton::Left )
            return true;
    }
    return false;
}

void MouseControlChecker::pluginEnabledChanged_( bool enabled )
{
    size_t connections = viewer->mouseDownSignal.num_slots();
    if ( enabled &&
         connections > mouseDownConnections_ &&
         checkUseLeftButton_() )
    {
        RibbonMenu* menu = viewer->getMenuPluginAs<RibbonMenu>().get();
        if ( menu )
            menu->pushNotification( {
                .text = "Camera operations that are controlled by left mouse button "
                        "may not work while this tool is active",
                .lifeTimeSec = 3.0f
            } );
    }
    mouseDownConnections_ = connections;
}

MRVIEWER_PLUGIN_REGISTRATION( MouseControlChecker )

}
