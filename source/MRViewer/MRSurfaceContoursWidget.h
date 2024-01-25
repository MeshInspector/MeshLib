#pragma once

#include "MRViewer.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRSurfacePointPicker.h"
#include "MRMesh/MRObjectMeshHolder.h"

#include <GLFW/glfw3.h>
#include <unordered_map>

namespace MR
{

class SurfaceContoursWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:

    struct SurfaceContoursWidgetParams {
        int widgetContourCloseMod = GLFW_MOD_CONTROL;
        int widgetDeletePointMod = GLFW_MOD_SHIFT;

    };


    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::ObjectMeshHolder> )>;
    using PickerPointObjectChecker = std::function<bool( std::shared_ptr<MR::ObjectMeshHolder> )>;

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, SurfaceContour>;

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    // enable or disable widget
    MRVIEWER_API void enable( bool isEnaled );

    // create a widget and connect it. 
    MRVIEWER_API void create( 
        PickerPointCallBack onPointAdd, 
        PickerPointCallBack onPointMove, 
        PickerPointCallBack onPointMoveFinish, 
        PickerPointCallBack onPointRemove,
        PickerPointObjectChecker isObjectValidToPick
    );

    // clear temp internal variables.
    MRVIEWER_API void clear();

    // reset widget, clear internal variables and detach from signals.
    MRVIEWER_API void reset();

    // return contour for specific object.
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::ObjectMeshHolder>& obj )
    {
        return pickedPoints_[obj];
    }

    // return all contours. 
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }

    // chech is contour closed for particular object.
    [[nodiscard]] bool isClosedCountour( const std::shared_ptr<ObjectMeshHolder>& obj );

    // shared variables. which need getters and setters.
    int activeIndex{ 0 };
    std::shared_ptr<MR::ObjectMeshHolder> activeObject = nullptr;


    // configuration params
    SurfaceContoursWidgetParams params;

private:

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& pt );

    // SurfaceContoursWidget interlal variables 
    bool moveClosedPoint_ = false;
    bool activeChange_ = false;
    bool isPickerActive_ = false;

    // data storage
    SurfaceContours pickedPoints_;

    // CallBack functions
    PickerPointCallBack onPointAdd_;
    PickerPointCallBack onPointMove_;
    PickerPointCallBack onPointMoveFinish_;
    PickerPointCallBack onPointRemove_;
    PickerPointObjectChecker isObjectValidToPick_;

    friend class AddPointActionPickerPoint;
    friend class RemovePointActionPickerPoint;
    friend class ChangePointActionPickerPoint;
};




}