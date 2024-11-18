#pragma once

#include "MRViewerEventsListener.h"
#include "MRViewport.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRSurfacePointPicker.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"

#include <unordered_map>
#include <unordered_set>

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

    // enable or disable widget
    MRVIEWER_API void enable( bool isEnabled );

    // create a widget and connect it.
    // To create a widget, you need to provide 4 callbacks and one function that determines whether this object can be used to place points.
    // All callback takes a shared pointer to an MR::VisualObject as an argument.
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

    /// clear temp internal variables.
    /// \param writeHistory - add history action (item in undo/redo). Set to false if you call the method as a part of another action.
    MRVIEWER_API void clear( bool writeHistory = true );

    // Reset widget, clear internal variables and detach from signals.
    // Also remove the undo/redo actions from the history.
    MRVIEWER_API void reset();

    // return contour for specific object, i.e. ordered vector of surface points
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::VisualObject>& obj )
    {
        return pickedPoints_[obj];
    }

    // return all contours, i.e. per object umap of ordered surface points [vestor].
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }

    // chech is contour closed for particular object.
    [[nodiscard]] bool isClosedCountour( const std::shared_ptr<VisualObject>& obj );

    // Correctly selects the last point in the contours.
    // If obj == nullptr then the check will be in all circuits.
    // If specified, then only in the contour on specyfied object
    void highlightLastPoint( const std::shared_ptr<VisualObject>& obj );

    // shared variables. which need getters and setters.
    MRVIEWER_API std::pair <std::shared_ptr<MR::VisualObject>, int > getActivePoint() const;
    MRVIEWER_API void setActivePoint( std::shared_ptr<MR::VisualObject> obj, int index );

    /// Get the active (the latest picked/moved) surface point widget.
    MRVIEWER_API std::shared_ptr<SurfacePointWidget> getActiveSurfacePoint() const;

    // Add a point to the end of non closed contour connected with obj.
    // With carefull it is possile to use it in CallBack.
    MRVIEWER_API bool appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint );

    // Remove point with pickedIndex index from contour connected with obj.
    // With carefull it is possile to use it in CallBack.
    MRVIEWER_API bool removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex );

    // Add a special transperent point contour to the end of contour connected with objectToCloseCoutour.
    // A coordinated of this special transperent point will be equal to the firs point in contour,
    // which will means that contour is closed.
    MRVIEWER_API bool closeContour( const std::shared_ptr<VisualObject>& objectToCloseCoutour );

    // configuration params
    SurfaceContoursWidgetParams params;
private:

    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt );

    // SurfaceContoursWidget interlal variables
    bool moveClosedPoint_ = false;
    bool activeChange_ = false;
    bool isPickerActive_ = false;

    // active point
    int activeIndex_{ 0 };
    std::shared_ptr<MR::VisualObject> activeObject_ = nullptr;

    // data storage
    SurfaceContours pickedPoints_;

    // picked points' cache
    std::unordered_set<const VisualObject*> surfacePointWidgetCache_;

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
    class AddPointActionPickerPoint : public WidgetHistoryAction
    {
    public:
        AddPointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point ) :
            widget_{ widget },
            obj_{ obj },
            point_{ point }
        {};

        virtual std::string name() const override;
        virtual void action( Type actionType ) override;
        [[nodiscard]] virtual size_t heapBytes() const override;
    private:
        SurfaceContoursWidget& widget_;
        const std::shared_ptr<MR::VisualObject> obj_;
        PickedPoint point_;
    };

    class RemovePointActionPickerPoint : public WidgetHistoryAction
    {
    public:
        RemovePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, int index ) :
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
        const std::shared_ptr<MR::VisualObject> obj_;
        PickedPoint point_;
        int index_;
    };

    class ChangePointActionPickerPoint : public WidgetHistoryAction
    {
    public:
        ChangePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, int index ) :
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
        const std::shared_ptr<MR::VisualObject> obj_;
        PickedPoint point_;
        int index_;
    };

    class SurfaceContoursWidgetClearAction : public WidgetHistoryAction
    {
    public:
        SurfaceContoursWidgetClearAction( std::string name, SurfaceContoursWidget& widget );

    public:
        [[nodiscard]] std::string name() const override { return name_; }

        void action( Type type ) override;

        [[nodiscard]] size_t heapBytes() const override;

    private:
        std::string name_;
        SurfaceContoursWidget& widget_;

        struct ObjectState
        {
            std::weak_ptr<VisualObject> objPtr;
            std::vector<PickedPoint> pickedPoints;
        };
        std::vector<ObjectState> states_;
        std::weak_ptr<VisualObject> activeObject_;
        int activeIndex_;
    };
};

}
