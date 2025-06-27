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

/// PickPointManager allows the user to pick/move/delete several ordered points on one or more visual objects;
/// mouse events and public methods automatically add history actions for reverting (if enabled)
class MRVIEWER_CLASS PickPointManager : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:
    using PickerPointCallBack = std::function<void( std::shared_ptr<VisualObject> obj, int index )>;
    using AllowCallBack = std::function<bool( const std::shared_ptr<VisualObject>& obj, int index )>;
    using ChangeObjectCallBack = std::function<bool( const std::shared_ptr<VisualObject>& obj )>;

    struct Params
    {
        /// Modifier key for closing a contour (ordered vector of points) using the widget
        int widgetContourCloseMod = GLFW_MOD_CONTROL;

        /// Modifier key for deleting a point using the widget
        int widgetDeletePointMod = GLFW_MOD_SHIFT;

        /// Whether to write undo history of all operations including public modifying functions and user actions
        bool writeHistory = true;

        /// This is appended to the names of all undo/redo actions
        std::string historyNameSuffix;

        /// Whether to activate dragging new point immediately after its creation on mouse down
        bool startDraggingJustAddedPoint = true;

        /// Parameters for configuring the surface point widget
        /// Parameters affect to future points only
        SurfacePointWidget::Parameters surfacePointParams;

        /// The color of all pick spheres except the one with the largest index on each object
        Color ordinaryPointColor = Color::gray();

        /// The color of last by index pick sphere in open contour
        Color lastPointColor = Color::green();

        /// The color of last by index pick sphere in closed contour, which coincides in position with the first pick sphere
        Color closeContourPointColor = Color::transparent();

        /// Predicate to additionally filter objects that should be treated as pickable.
        Viewport::PickRenderObjectPredicate pickPredicate;

        /// This callback is invoked before addition of new point (with index=-1) by mouse (but not from API or history),
        /// the addition is canceled if this callback returns false
        AllowCallBack canAddPoint;

        /// This callback is invoked after a point is added with its index
        PickerPointCallBack onPointAdd;

        /// This callback is invoked when a point starts being dragged
        PickerPointCallBack onPointMoveStart;

        /// This callback is invoked every time after currently dragged point is moved (in between onPointMoveStart and onPointMoveFinish)
        PickerPointCallBack onPointMove;

        /// This callback is invoked when point's dragging is completed
        PickerPointCallBack onPointMoveFinish;

        /// This callback is invoked before removal of some point by mouse (but not from API or history),
        /// the removal is canceled if this callback returns false
        AllowCallBack canRemovePoint;

        /// This callback is invoked when a point is removed with its index before deletion
        PickerPointCallBack onPointRemove;

        /// This callback is invoked when an object was changed and needed update of points
        /// Return false if need to skip internal updates
        ChangeObjectCallBack onUpdatePoints;

    };
    Params params;

    /// A common base class for all history actions of this widget.
    struct WidgetHistoryAction : HistoryAction {};

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<VisualObject>, SurfaceContour>;

    /// create an object and starts listening for mouse events
    MRVIEWER_API PickPointManager();

    /// destroy this and remove the undo/redo actions referring this from the history.
    MRVIEWER_API ~PickPointManager();

    /// return contour for specific object (creating new one if necessary)
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<VisualObject>& obj ) { return pickedPoints_[obj]; }

    /// return all contours, i.e. per object unorderd_map of ordered surface points [vector].
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const { return pickedPoints_; }

    /// check whether the contour is closed for a particular object.
    [[nodiscard]] MRVIEWER_API bool isClosedContour( const std::shared_ptr<VisualObject>& obj ) const;

    /// returns the total number of pick points (including extra point if the contour is closed) on given object
    [[nodiscard]] MRVIEWER_API size_t numPickPoints( const std::shared_ptr<VisualObject>& obj ) const;

    /// returns point widget by index from given object or nullptr if no such widget exists
    [[nodiscard]] MRVIEWER_API std::shared_ptr<SurfacePointWidget> getPointWidget( const std::shared_ptr<VisualObject>& obj, int index ) const;

    /// returns index of given point widget on given object or -1 if this widget is not from given object
    [[nodiscard]] MRVIEWER_API int getPointIndex( const std::shared_ptr<VisualObject>& obj, SurfacePointWidget& pointWidget ) const;

    /// returns point widget currently dragged by mouse
    [[nodiscard]] SurfacePointWidget* draggedPointWidget() const { return draggedPointWidget_; }

    /// Add a point to the end of non closed contour connected with obj
    /// \param startDragging if true then new point widget is immediately made draggable by mouse, please be sure that mouse is over new point and is down
    MRVIEWER_API bool appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint, bool startDragging = false );

    /// Inserts a point into contour connected with obj
    /// \param index point index before which to insert new point
    /// \param startDragging if true then new point widget is immediately made draggable by mouse, please be sure that mouse is over new point and is down
    MRVIEWER_API bool insertPoint( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& triPoint, bool startDragging = false );

    /// Remove point with pickedIndex index from contour connected with obj.
    MRVIEWER_API bool removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex );

    // if ( makeClosed ), and the contour is open add a special transparent point contour to the end of contour connected with given object.
    // A coordinated of this special transparent point will be equal to the firs point in contour, which will means that contour is closed.
    // if ( !makeClosed ), and the contour is closed remove last point of contour connected with given object.
    MRVIEWER_API bool closeContour( const std::shared_ptr<VisualObject>& obj, bool makeClosed = true );

    struct ObjectState
    {
        std::weak_ptr<VisualObject> objPtr;
        std::vector<PickedPoint> pickedPoints;
    };
    using FullState = std::vector<ObjectState>;

    /// returns the state of this
    MRVIEWER_API FullState getFullState() const;

    /// removes all points from all objects
    MRVIEWER_API void clear();

    /// removes all current points, then adds pick points on all objects as prescribed by given state
    MRVIEWER_API void setFullState( FullState s );

private:
    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    ObjAndPick pick_() const;

    /// sets the color of last and pre-last pick spheres for given object;
    /// the colors are taken from parameters
    void colorLast2Points_( const std::shared_ptr<VisualObject>& obj );

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<VisualObject>& obj, const PickedPoint& pt );

    /// removes everything
    void clearNoHistory_();

    /// adds pick points on all objects as prescribed by given state,
    /// puts original state of this in s on return
    void swapStateNoHistory_( FullState& s );

    /// \param index point index before which to insert new point, -1 here means insert after last one
    /// \return index of just inserted point
    int insertPointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& point, bool startDragging );

    /// \return location of just removed point
    PickedPoint removePointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index );

    /// if history writing is enabled, constructs history action and appends it to global store
    template<class HistoryActionType, typename... Args>
    void appendHistory_( Args&&... args );

    /// if history writing is enabled, appends given history action to global store
    void appendHistory_( std::shared_ptr<HistoryAction> action ) const;

    /// setup new hovered point widget, and removes hovering from the previous one
    void setHoveredPointWidget_( SurfacePointWidget* newHoveredPoint );

    // whether the contour was closed before dragging of point #0, so we need to move the last point on end drag
    bool moveClosedPoint_ = false;

    // point widget currently under mouse and highlighted
    SurfacePointWidget* hoveredPointWidget_ = nullptr;

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
    class SetStateHistoryAction;
    class AddRemovePointHistoryAction;
    class MovePointHistoryAction;
};

} //namespace MR
