#pragma once

#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRMatrix3.h"

namespace MR
{

/// Widget for visualizing a ray or axis (origin point and direction)
/// Based on DirectionWidget
class RayWidget : public MultiListener<DragStartListener, DragListener, DragEndListener, MouseClickListener>
{
public:
    /// This callback is invoked every time when the ray is changed by mouse
    using OnRayChangedCallback = std::function<void( const Line3f& )>;

private:
    // An object that is first part cylinder centered around 0 and second part sphere in 0
    std::shared_ptr<ObjectMesh> lineObj_;
    size_t cylinderFaces_ = 0;

    OnRayChangedCallback onRayChanged_;

    bool pickingPoint_ = false;

    Line3f line_;
    float length_ = 0.0f;
    bool settingDir_ = false;
    bool pressed_ = false; // this widget accepted the current drag start
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
    void create( const Line3f& ray, float length, OnRayChangedCallback onRayChanged );

    /// Removes the widget from the scene
    /// unsubscribes from viewer events
    void reset();

    /// Updates the ray
    void updateRay( const Line3f& ray );
    void updateDirection( const Vector3f& dir );
    void updateOrigin( const Vector3f& org );

    /// Updates the length of the visual object
    void updateLength( float length );

    /// Sets the visibility of the widget
    void setVisible( bool visible );

    /// Pick point mode
    void setPickMode( bool pickMode );
    bool isPickMode() const { return pickingPoint_; }

    /// Returns the ray origin
    const Vector3f& getOrg() const { return line_.p; }
private:
    virtual bool onDragStart_( MouseButton button, int modifiers ) override;
    virtual bool onDrag_( int x, int y ) override;
    virtual bool onDragEnd_( MouseButton button, int modifiers ) override;
    virtual bool onMouseClick_( MouseButton button, int modifiers ) override;
};

}
