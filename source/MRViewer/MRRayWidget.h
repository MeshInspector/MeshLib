#pragma once

#include "exports.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{

/// Widget for visualizing a ray or axis (origin point and direction)
/// Based on DirectionWidget
class MRVIEWER_CLASS RayWidget : public MultiListener<DragStartListener, DragListener, DragEndListener, MouseClickListener>
{
public:
    /// This callback is invoked every time when the ray is changed by mouse
    using OnRayChangedCallback = std::function<void( const Line3f&, bool )>;

    /// history action for changing the ray. It should be added to the history stack by user code
    class ChangeRayAction : public ChangeXfAction
    {
    public:
        ChangeRayAction( RayWidget& widget ) :
            ChangeXfAction( "Change Axis", static_pointer_cast<Object>( widget.lineObj_ ) ),
            widget_{ widget },
            line_{ widget.line_ }
        {}
        virtual void action( Type type ) override
        {
            ChangeXfAction::action( type );
            auto tempDir = widget_.line_;
            widget_.updateRay( line_ );
            line_ = tempDir;
        }
    private:
        RayWidget& widget_;
        Line3f line_;
    };

    /// history action for changing the length. It should be added to the history stack by user code
    class ChangeLengthAction : public ChangeXfAction
    {
    public:
        ChangeLengthAction( RayWidget& widget ) :
            ChangeXfAction( "Change Axis Length", static_pointer_cast<Object>( widget.lineObj_ ) ),
            widget_{ widget },
            length_{ widget.length_ }
        {}
        virtual void action( Type ) override
        {
            auto len = widget_.length_;
            widget_.updateLength( length_ );
            length_ = len;

        }
    private:
        RayWidget& widget_;
        float length_;
    };

    /// history action for changing the visible. It should be added to the history stack by user code
    class ChangeVisibleAction : public HistoryAction
    {
    public:
        ChangeVisibleAction( RayWidget& widget ) :
            widget_{ widget },
            visible_{ widget.lineObj_->visibilityMask() }
        {}
        virtual void action( Type ) override
        {
            auto oldVisible = widget_.lineObj_->visibilityMask();
            widget_.lineObj_->setVisibilityMask( visible_ );
            visible_ = oldVisible;
        }
        virtual std::string name() const override
        {
            return name_;
        }
        [[nodiscard]] virtual size_t heapBytes() const override
        {
            return name_.capacity();
        }
    private:
        RayWidget& widget_;
        ViewportMask visible_;
        std::string name_ = "Change Visible";
    };

private:
    // An object that is first part cylinder centered around 0 and secong part sphere in 0
    std::shared_ptr<ObjectMesh> lineObj_;
    size_t cylinderFaces_;
    VisualObject* parent_;


    bool active_ = false;
    OnRayChangedCallback onRayChanged_;
    Color color_ = Color::green();
    bool needToSaveHistory_ = true;

    bool pickingPoint_ = false;

    Line3f line_;
    float length_;
    bool settingDir_ = false;
    // When blocked, cannot be moved with mouse
    bool blockedMouse_{ false };
    // Dragging parameters
    Vector3f worldStartOrg_;
    Vector3f worldStartPoint_;
    Matrix3f startDirRotation_;
    float viewportStartPointZ_{ 0.0f };

    void createObject_();
    void clearObject_();
    void updateXf_();

public:
    /// Creates a new widget for visualizing the ray/axis and adds it to scene
    /// subscribes to viewer events
    /// @param ray initial ray
    /// @param length length of the visual object
    /// @param onRayChanged callback for the ray change
    /// @param parent parent object for the widget, nullptr means scene root
    MRVIEWER_API void create( const Line3f& ray, float length, OnRayChangedCallback onRayChanged, VisualObject* parent = nullptr  );

    /// Removes the widget from the scene
    /// unsubscribes from viewer events
    MRVIEWER_API void reset();

    /// Check if the object is active (after create(), before reset())
    bool isActive() const { return active_; }

    /// Manually set callback function
    MRVIEWER_API void setOnRayChangedCallback( OnRayChangedCallback cb );

    /// Updates the ray
    MRVIEWER_API void updateRay( const Line3f& ray );
    MRVIEWER_API void updateDirection( const Vector3f& dir );
    MRVIEWER_API void updateOrigin( const Vector3f& org );

    /// Updates the length of the visual object
    MRVIEWER_API void updateLength( float length );

    /// Returns internal data model object of this widget
    std::shared_ptr<ObjectMesh> obj() const { return lineObj_; }

    /// Sets the visibility of the widget
    MRVIEWER_API void setVisible( bool visible );
    MRVIEWER_API bool isVisible() const;

    /// Pick point mode
    MRVIEWER_API void setPickMode( bool pickMode );
    bool isPickMode() const { return pickingPoint_; }

    /// Sets the color of the widget
    MRVIEWER_API void setColor( const Color& color );

    /// Returns the color of the widget
    MRVIEWER_API const Color& getColor() const;

    /// Returns the ray
    const Line3f& getRay() const { return line_; }
    const Vector3f& getOrg() const { return line_.p; }
    const Vector3f& getDirection() const { return line_.d; }

    /// Returns pointer to parent object
    const VisualObject* getParentPtr() const { return parent_; }

    /// Block or allow mouse editing (allowed by default)
    bool isMouseBlocked() const { return blockedMouse_; }

    void setMouseBlocked( bool blocked ) { blockedMouse_ = blocked; }
private:
    MRVIEWER_API virtual bool onDragStart_( MouseButton button, int modifiers ) override;
    MRVIEWER_API virtual bool onDragEnd_( MouseButton button, int modifiers ) override;
    MRVIEWER_API virtual bool onDrag_( int x, int y ) override;
    MRVIEWER_API virtual bool onMouseClick_( MouseButton button, int modifiers ) override;
};

}
