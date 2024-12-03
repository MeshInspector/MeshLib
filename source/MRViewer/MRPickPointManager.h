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

/// this object allows the user to pick/move/delete several ordered points on one or more visual objects
class MRVIEWER_CLASS PickPointManager : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:
    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::VisualObject> obj, int index )>;

    struct Params
    {
        /// Modifier key for closing a contour (ordered vector of points) using the widget
        int widgetContourCloseMod = GLFW_MOD_CONTROL;

        /// Modifier key for deleting a point using the widget
        int widgetDeletePointMod = GLFW_MOD_SHIFT;

        /// Indicates whether to write history of the contours
        bool writeHistory = true;

        /// This is appended to the names of all undo/redo actions.
        std::string historyNameSuffix;

        /// Parameters for configuring the surface point widget
        /// Parameters affect to future points only
        SurfacePointWidget::Parameters surfacePointParams;

        /// Color for ordinary points in the contour
        /// Parameters affect to future points only
        MR::Color ordinaryPointColor = Color::gray();

        /// Color for the last modified point in the contour
        /// Parameters affect to future points only
        MR::Color lastPointColor = Color::green();

        /// Color for the special point used to close a contour. Better do not change it.
        /// Parameters affect to future points only
        MR::Color closeContourPointColor = Color::transparent();

        /// Predicate to additionally filter objects that should be treated as pickable.
        Viewport::PickRenderObjectPredicate pickPredicate;

        /// This callback is invoked after a point is added with its index.
        PickerPointCallBack onPointAdd;

        /// This callback is invoked when a point starts being dragged.
        PickerPointCallBack onPointMoveStart;

        /// This callback is invoked when point's dragging is completed.
        PickerPointCallBack onPointMoveFinish;

        /// This callback is invoked when a point is removed with its index before deletion.
        PickerPointCallBack onPointRemove;
    };
    Params params;

    // A common base class for all history actions of this widget.
    struct WidgetHistoryAction : HistoryAction {};

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::VisualObject>, SurfaceContour>;

    /// create an object and starts listening for mouse events
    MRVIEWER_API PickPointManager();

    /// destroy this and remove the undo/redo actions from the history.
    MRVIEWER_API ~PickPointManager();

    // return contour for specific object (creating new one if necessary)
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

    /// returns point widget by index from given object or nullptr if no such widget exists
    [[nodiscard]] MRVIEWER_API std::shared_ptr<SurfacePointWidget> getPointWidget( const std::shared_ptr<VisualObject>& obj, int index ) const;

    /// returns point widget currently dragged by mouse
    [[nodiscard]] SurfacePointWidget* draggedPointWidget() const { return draggedPointWidget_; }

    // Add a point to the end of non closed contour connected with obj.
    MRVIEWER_API bool appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint );

    // Remove point with pickedIndex index from contour connected with obj.
    MRVIEWER_API bool removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex );

    /// remove all points from all objects
    /// \param writeHistory - add history action (item in undo/redo). Set to false if you call the method as a part of another action.
    MRVIEWER_API void clear( bool writeHistory = true );

    // if ( makeClosed ), and the contour is open add a special transparent point contour to the end of contour connected with given object.
    // A coordinated of this special transparent point will be equal to the firs point in contour, which will means that contour is closed.
    // if ( !makeClosed ), and the contour is closed remove last point of contour connected with given object.
    MRVIEWER_API bool closeContour( const std::shared_ptr<VisualObject>& obj, bool makeClosed = true );

private:
    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    ObjAndPick pick_() const;

    /// sets the color of last and pre-last pick spheres for given object;
    /// the colors are taken from parameters
    void colorLast2Points_( const std::shared_ptr<VisualObject>& obj );

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt );

    /// \param index point index before which to insert new point, -1 here means insert after last one
    /// \return index of just inserted point
    int insertPointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& point );

    /// \return location of just removed point
    PickedPoint removePointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index );

    // whether the contour was closed before dragging of point #0, so we need to move the last point on end drag
    bool moveClosedPoint_ = false;

    // point widget currently dragged by mouse
    SurfacePointWidget* draggedPointWidget_ = nullptr;

    // data storage
    SurfaceContours pickedPoints_;

    // all spheres created in createPickWidget_ to quickly differentiate them from other features
    HashSet<const VisualObject*> myPickSpheres_;

    // for each object with pick points, holds connections to mesh/point changed event
    struct ConnectionHolder
    {
        boost::signals2::scoped_connection onMeshChanged;
        boost::signals2::scoped_connection onPointsChanged;
    };
    HashMap<std::shared_ptr<VisualObject>, ConnectionHolder> connectionHolders_;

    // History classes:
    class AddRemovePointHistoryAction;
    class MovePointHistoryAction;
    class ClearHistoryAction;
};

} //namespace MR
