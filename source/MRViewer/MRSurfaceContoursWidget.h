#pragma once

#include "MRViewer.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRSurfacePointPicker.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"
#include <unordered_map>

namespace MR
{

class SurfaceContoursWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:

    struct SurfaceContoursWidgetParams {
        // Modifier key for closing a contour (ordered vector of points) using the widget
        int widgetContourCloseMod = GLFW_MOD_CONTROL;

        // Modifier key for deleting a point using the widget
        int widgetDeletePointMod = GLFW_MOD_SHIFT;

        // Indicates whether to write history of the contours
        bool writeHistory = true;

        // Indicates whether to flash history at the end of the operation
        bool filterHistoryonReset = true;

        // Parameters for configuring the surface point widget
        // Parameters affect to future points only, if need update existing one use void updateAllPointsWidgetParams( const SurfacePointWidget::Parameters& p )
        SurfacePointWidget::Parameters surfacePointParams;

        // Color for ordinary points in the contour
        MR::Color ordinaryPointColor = Color::gray();

        // Color for the last modified point in the contour
        MR::Color lastPoitColor = Color::green();

        // Color for the special point used to close a contour. Better do not change it. 
        MR::Color closeContourPointColor = Color::transparent();
    };

    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::ObjectMeshHolder> )>;
    using PickerPointObjectChecker = std::function<bool( std::shared_ptr<MR::ObjectMeshHolder> )>;

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, SurfaceContour>;

    // enable or disable widget
    MRVIEWER_API void enable( bool isEnaled );

    // create a widget and connect it. 
    // To create a widget, you need to provide 4 callbacks and one function that determines whether this object can be used to place points.
    // All callback takes a shared pointer to an MR::ObjectMeshHolder as an argument.
    // onPointAdd: This callback is invoked when a point is added.
    // onPointMove : This callback is triggered when a point is being start  moved or dragge.
    // onPointMoveFinish : This callback is called when the movement of a point is completed.
    // onPointRemove : This callback is executed when a point is removed.
    // isObjectValidToPick : Must returh true or false. This callback is used to determine whether an object is valid for picking.
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

    // return contour for specific object, i.e. ordered vector of surface points
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::ObjectMeshHolder>& obj )
    {
        return pickedPoints_[obj];
    }

    // return all contours, i.e. per object umap of ordered surface points [vestor].
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }

    // chech is contour closed for particular object.
    [[nodiscard]] bool isClosedCountour( const std::shared_ptr<ObjectMeshHolder>& obj );

    // updates the parameters of all existing points ( SurfacePointWidget ) in the contours, and also sets their points that will be created later
    void updateAllPointsWidgetParams( const SurfacePointWidget::Parameters& p );

    // shared variables. which need getters and setters.
    int activeIndex{ 0 };
    std::shared_ptr<MR::ObjectMeshHolder> activeObject = nullptr;

    // configuration params
    SurfaceContoursWidgetParams params;
private:

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

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


// History classes;
class AddPointActionPickerPoint : public HistoryAction
{
public:
    AddPointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point }
    {};

    virtual std::string name() const override;
    virtual void action( Type actionType ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
};

class RemovePointActionPickerPoint : public HistoryAction
{
public:
    RemovePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override;
    virtual void action( Type actionType ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};

class ChangePointActionPickerPoint : public HistoryAction
{
public:
    ChangePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override;
    virtual void action( Type ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};



}