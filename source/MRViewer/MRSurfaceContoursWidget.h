#pragma once

#include "MRViewerEventsListener.h"
#include "MRViewport.h"
#include "MRSurfacePointPicker.h"
#include "MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include <MRMesh/MRphmap.h>
#include <unordered_map>

namespace MR
{

class MRVIEWER_CLASS SurfaceContoursWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:
    struct SurfaceContoursWidgetParams
    {
        // Modifier key for closing a contour (ordered vector of points) using the widget
        int widgetContourCloseMod = GLFW_MOD_CONTROL;

        // Modifier key for deleting a point using the widget
        int widgetDeletePointMod = GLFW_MOD_SHIFT;

        // Indicates whether to write history of the contours
        bool writeHistory = true;

        // This is appended to the names of all undo/redo actions.
        std::string historyNameSuffix;

        // Indicates whether to flash history on reset call
        bool filterHistoryonReset = true;

        // Parameters for configuring the surface point widget
        // Parameters affect to future points only
        SurfacePointWidget::Parameters surfacePointParams;

        // Color for ordinary points in the contour
        // Parameters affect to future points only
        MR::Color ordinaryPointColor = Color::gray();

        // Color for the last modified point in the contour
        // Parameters affect to future points only
        MR::Color lastPoitColor = Color::green();

        // Color for the special point used to close a contour. Better do not change it.
        // Parameters affect to future points only
        MR::Color closeContourPointColor = Color::transparent();

        // Predicate to additionally filter objects that should be treated as pickable.
        Viewport::PickRenderObjectPredicate pickPredicate;
    };

    // A common base class for all history actions of this widget.
    struct WidgetHistoryAction : HistoryAction {};

    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::VisualObject> )>;
    using PickerPointObjectChecker = std::function<bool( std::shared_ptr<MR::VisualObject> )>;

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::VisualObject>, SurfaceContour>;

    // create a widget and connect it.
    // To create a widget, you need to provide 4 callbacks and one function that determines whether this object can be used to place points.
    // All callback takes a shared pointer to an MR::VisualObject as an argument.
    // onPointAdd: This callback is invoked when a point is added.
    // onPointMove : This callback is triggered when a point is being start  moved or dragged.
    // onPointMoveFinish : This callback is called when the movement of a point is completed.
    // onPointRemove : This callback is executed when a point is removed.
    // isObjectValidToPick : Must return true or false. This callback is used to determine whether an object is valid for picking.
    MRVIEWER_API SurfaceContoursWidget(
            PickerPointCallBack onPointAdd,
            PickerPointCallBack onPointMove,
            PickerPointCallBack onPointMoveFinish,
            PickerPointCallBack onPointRemove,
            PickerPointObjectChecker isObjectValidToPick
    );

    // Also remove the undo/redo actions from the history.
    MRVIEWER_API ~SurfaceContoursWidget();

    /// clear temp internal variables.
    /// \param writeHistory - add history action (item in undo/redo). Set to false if you call the method as a part of another action.
    MRVIEWER_API void clear( bool writeHistory = true );

    // return contour for specific object, i.e. ordered vector of surface points
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::VisualObject>& obj )
    {
        return pickedPoints_[obj];
    }

    // return all contours, i.e. per object unorderd_map of ordered surface points [vector].
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }

    // check whether the contour is closed for a particular object.
    [[nodiscard]] MRVIEWER_API bool isClosedCountour( const std::shared_ptr<VisualObject>& obj ) const;

    // Correctly selects the last point in the contours.
    // If obj == nullptr then the check will be in all circuits.
    // If specified, then only in the contour on specified object
    MRVIEWER_API void highlightLastPoint( const std::shared_ptr<VisualObject>& obj );

    // shared variables. which need getters and setters.
    [[nodiscard]] MRVIEWER_API std::pair <std::shared_ptr<MR::VisualObject>, int > getActivePoint() const;
    MRVIEWER_API void setActivePoint( std::shared_ptr<MR::VisualObject> obj, int index );

    /// Get the active (the latest picked/moved) surface point widget.
    [[nodiscard]] MRVIEWER_API std::shared_ptr<SurfacePointWidget> getActiveSurfacePoint() const;

    // Add a point to the end of non closed contour connected with obj.
    MRVIEWER_API bool appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint );

    // Remove point with pickedIndex index from contour connected with obj.
    MRVIEWER_API bool removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex );

    // if ( makeClosed ), and the contour is open add a special transparent point contour to the end of contour connected with given object.
    // A coordinated of this special transparent point will be equal to the firs point in contour, which will means that contour is closed.
    // if ( !makeClosed ), and the contour is closed remove last point of contour connected with given object.
    MRVIEWER_API bool closeContour( const std::shared_ptr<VisualObject>& obj, bool makeClosed = true );

    // configuration params
    SurfaceContoursWidgetParams params;

private:
    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    ObjAndPick pick_() const;

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt );

    // SurfaceContoursWidget internal variables
    bool moveClosedPoint_ = false;

    // pick sphere currently dragged by mouse
    const VisualObject * draggedPickSphere_ = nullptr;

    // active point
    int activeIndex_{ 0 };
    std::shared_ptr<MR::VisualObject> activeObject_ = nullptr;

    // data storage
    SurfaceContours pickedPoints_;

    // all spheres created in createPickWidget_ to quickly differentiate them from other features
    HashSet<const VisualObject*> myPickSpheres_;

    // connection storage
    struct SurfaceConnectionHolder
    {
        boost::signals2::scoped_connection onMeshChanged;
        boost::signals2::scoped_connection onPointsChanged;
    };
    std::unordered_map<std::shared_ptr<VisualObject>, SurfaceConnectionHolder> surfaceConnectionHolders_;

    // CallBack functions
    PickerPointCallBack onPointAdd_;
    PickerPointCallBack onPointMove_;
    PickerPointCallBack onPointMoveFinish_;
    PickerPointCallBack onPointRemove_;
    PickerPointObjectChecker isObjectValidToPick_;

    // undo/redo flag; used by the history action classes to force disable internal checks
    bool undoRedoMode_{ false };

    // History classes:
    class AddRemovePointHistoryAction;
    class ChangePointActionPickerPoint;
    class SurfaceContoursWidgetClearAction;
};

} //namespace MR
