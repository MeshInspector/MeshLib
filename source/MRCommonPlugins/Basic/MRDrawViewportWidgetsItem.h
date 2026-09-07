#pragma once

#include "MRCommonPlugins/exports.h"
#include "MRViewer/MRRibbonMenuItem.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MRSignal.h"

#include <boost/signals2/connection.hpp>

namespace MR
{

// Inherit your plugin from this to draw viewport widgets from there.
class ProvidesViewportWidget
{
public:
    MRCOMMONPLUGINS_API ProvidesViewportWidget();
    ProvidesViewportWidget( const ProvidesViewportWidget& ) = delete;
    ProvidesViewportWidget& operator=( const ProvidesViewportWidget& ) = delete;
    virtual ~ProvidesViewportWidget() = default; // Don't strictly need this, but MSVC warns otherwise.


    struct ViewportWidgetInterface
    {
        virtual ~ViewportWidgetInterface() = default; // Don't strictly need this, but MSVC warns otherwise.

        // Which viewport we're dealing with.
        [[nodiscard]] virtual ViewportId viewportId() const = 0;

        // Register a new button. They will be sorted by `order` (then by name) ascending, left-to-right.
        // The `onClick` callback will be called late, make sure it doens't dangle the captures.
        // `active` only affects how the button is rendered.
        virtual void addButton( float order, std::string name, bool active, std::string icon, std::string tooltip, std::function<void()> onClick ) = 0;
    };

    virtual void providedViewportWidgets( ViewportWidgetInterface& in ) = 0;

private:
    boost::signals2::scoped_connection providedWidgetsConnection_;
};

// This sits in the background and renders various per-viewport buttons.
class DrawViewportWidgetsItem : public RibbonMenuItem
{
public:
    MRCOMMONPLUGINS_API DrawViewportWidgetsItem();

    bool action() override { return false; }

private:
    void handleViewport( Viewport& viewport );

    boost::signals2::scoped_connection preDrawConnection_;

    using HandleViewportSignal = boost::signals2::signal<void( ProvidesViewportWidget::ViewportWidgetInterface& in )>;
    static HandleViewportSignal& getHandleViewportSignal_();

    friend ProvidesViewportWidget; // To let it access `handleViewportSignal_`.
};

}
