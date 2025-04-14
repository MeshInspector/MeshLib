#pragma once

#include "exports.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{
/// Widget for visualizing the direction
class MRVIEWER_CLASS DirectionWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    /// This callback is invoked every time when the direction is changed by mouse
    using OnDirectionChangedCallback = std::function<void( const Vector3f&, bool )>;

    /// This history action must be created before the change in widget's direction or base to make them undo-able
    class ChangeDirAction : public ChangeXfAction
    {
    public:
        ChangeDirAction( DirectionWidget& widget, const std::string& name = "Change Direction" ) :
            ChangeXfAction( name, static_pointer_cast<Object>( widget.directionObj_ ) )
        {}
    };

    /// history action for changing the length. It should be added to the history stack by user code
    class ChangeLengthAction : public ChangeXfAction
    {
    public:
        ChangeLengthAction( DirectionWidget& widget ) :
            ChangeXfAction( "Change Length", static_pointer_cast< Object >( widget.directionObj_ ) ),
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
        DirectionWidget& widget_;
        float length_;
    };

    /// history action for changing the visible. It should be added to the history stack by user code
    class ChangeVisibleAction : public HistoryAction
    {
    public:
        ChangeVisibleAction( DirectionWidget& widget ) :
            widget_{ widget },
            visible_{ widget.directionObj_->visibilityMask() }
        {}
        virtual void action( Type ) override
        {
            auto oldVisible = widget_.directionObj_->visibilityMask();
            widget_.directionObj_->setVisibilityMask( visible_ );
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
        DirectionWidget& widget_;
        ViewportMask visible_;
        std::string name_ = "Change Visible";
    };

private:
    std::shared_ptr<ObjectMesh> directionObj_;

    float length_ = 0;
    bool mousePressed_ = false;
    // if blocked cannot be moved with mouse
    bool blockedMouse_{ false };
    Vector3f worldStartPoint_;
    float viewportStartPointZ_{ 0.0f };
    OnDirectionChangedCallback onDirectionChanged_;
    Color color_ = Color::red();
    bool needToSaveHistory_ = true;
    void clear_();

public:
    /// Creates a new widget for visualizing the direction and adds it to scene
    /// subscribes to viewer events
    /// @param dir initial direction
    /// @param base initial base of the arrow
    /// @param length length of the arrow
    /// @param onDirectionChanged callback for the direction change
    /// @param parent parent object for the widget, nullptr means scene root
    MRVIEWER_API void create( const Vector3f& dir, const Vector3f& base, float length, OnDirectionChangedCallback onDirectionChanged, Object* parent = nullptr  );

    /// Removes the widget from the scene
    /// unsubscribes from viewer events
    MRVIEWER_API void reset();

    /// Manually set callback function
    MRVIEWER_API void setOnDirectionChangedCallback( OnDirectionChangedCallback cb );

    /// Updates the direction of the arrow, in world space
    MRVIEWER_API void updateDirection( const Vector3f& dir );

    /// Updates the direction of the arrow in parent's space
    MRVIEWER_API void updateLocalDirection( const Vector3f& dir );

    /// Updates the base of the arrow, in world space
    MRVIEWER_API void updateBase( const Vector3f& base );

    /// Updates the base of the arrow in parent's space
    MRVIEWER_API void updateLocalBase( const Vector3f& base );

    /// Updates the length of the arrow
    MRVIEWER_API void updateLength( float length );

    /// Updates the base and the length of the arrow
    MRVIEWER_API void updateArrow( const Vector3f& base, float length );
    
    /// Returns internal data model object of this widget
    std::shared_ptr<ObjectMesh> obj() const { return directionObj_; }

    /// Sets the visibility of the widget
    MRVIEWER_API void setVisible( bool visible );

    MRVIEWER_API bool isVisible() const;

    /// Sets the color of the widget
    MRVIEWER_API void setColor( const Color& color );

    /// Returns the color of the widget
    MRVIEWER_API const Color& getColor() const;

    /// Returns the base of the widget, in world space
    MRVIEWER_API Vector3f getBase() const;

    /// Returns the base of the widget in parent's space
    MRVIEWER_API Vector3f getLocalBase() const;

    /// Returns the direction of the widget, in world space
    MRVIEWER_API Vector3f getDirection() const;

    /// Returns the direction of the widget in parent's space
    MRVIEWER_API Vector3f getLocalDirection() const;

    /// Returns pointer to parent object, always not-null after create() and before reset()
    MRVIEWER_API Object* getParentPtr() const;

    /// Block or allow mouse editing (allowed by default)
    bool isMouseBlocked() const { return blockedMouse_; }

    void setMouseBlocked( bool blocked ) { blockedMouse_ = blocked; }
private:
    MRVIEWER_API virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
};

} //namespace MR
