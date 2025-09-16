#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRViewportGlobalBasis.h"
#include "MRCommonPlugins/Basic/MRDrawViewportWidgetsItem.h"

namespace MR
{

class ToggleProjectionModeItem : public RibbonMenuItem, public ProvidesViewportWidget
{
public:
    ToggleProjectionModeItem();
    bool action() override { return false; }

    void providedViewportWidgets( ViewportWidgetInterface& in ) override;

    ViewportMask showButtonInViewports = ViewportMask::all();
};

ToggleProjectionModeItem::ToggleProjectionModeItem() :
    RibbonMenuItem( "Toggle Projection Mode" )
{}

void ToggleProjectionModeItem::providedViewportWidgets( ViewportWidgetInterface& in )
{
    auto id = in.viewportId();
    if ( !showButtonInViewports.contains( id ) )
        return;

    bool isOrtho = getViewerInstance().viewport( id ).getParameters().orthographic;

    in.addButton( 30, "Projection", false, isOrtho ? "Viewport projection orthographic" : "Viewport projection perspective",
        isOrtho ? "Projection: switch to perspective" : "Projection: switch to orthographic",
        [id]
        {
            auto& viewport = getViewerInstance().viewport( id );
            viewport.setOrthographic( !viewport.getParameters().orthographic );
        }
    );
}

MR_REGISTER_RIBBON_ITEM( ToggleProjectionModeItem )

}
