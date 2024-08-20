#pragma once
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRViewer/MRViewerFwd.h"
#include "MRViewer/MRMouse.h"
#include "imgui.h"

namespace MR
{

/// Helper class to incorporate basic object transformation feature into plugins
/// User can move objects by dragging them, rotate by dragging with Ctrl key; scaling is disabled by default
/// To use, create class instance and call its event handlers
/// For extra features, override the `pick_` method
class MRVIEWER_CLASS MoveObjectByMouseImpl
{
public:
    MoveObjectByMouseImpl() = default;
    virtual ~MoveObjectByMouseImpl() = default;

    /// Minimum drag distance in screen pixels
    /// If cursor moved less than this value, no transform is done
    /// Default value 0 (set transformation even if mouse did not move)
    int minDistance() const { return minDistance_; }
    void setMinDistance( int minDistance ) { minDistance_ = minDistance; }

    /// Drawing callback to draw lines and tooltips
    /// Should be called from `drawDialog`
    MRVIEWER_API void onDrawDialog( float menuScaling ) const;

    /// These functions should be called from corresponding mouse handlers
    /// Or mouse drag handlers, making it work together with mouseClick signal
    /// Return true if handled (picked/moved/released)
    /// It is recommended to call `viewer->select_hovered_viewport()` before `onMouseDown`
    /// The object is transformed temporarily in `onMouseMove`
    /// `onMouseUp` finalizes the transformation and writes to undo/redo history (unless cancelled)
    MRVIEWER_API bool onMouseDown( MouseButton button, int modifiers );
    MRVIEWER_API bool onMouseMove( int x, int y );
    MRVIEWER_API bool onMouseUp( MouseButton button, int modifiers );

    /// Returns true if currently moving object(s)
    /// Return false if not active, or object picked but `minDistance` has not yet reached
    MRVIEWER_API bool isMoving() const;

    /// Reset transformation and stop moving the object(s). Does nothing if not moving anything
    /// Calling `onMouseUp` is not necessary after this
    /// Should be called when closing plugin etc.
    MRVIEWER_API void cancel();

protected:
    /// Transformation mode
    enum class TransformMode
    {
        None,
        Translation,
        Rotation,
        Scale
    };

    /// Called from `onMouseDown`
    /// Returns chosen `TransformMode` to start transformation, `None` to skip; fills the output parameters:
    /// `objects` - list of objects to be affected by transformation
    /// `centerPoint` - center point for rotation (world coordinates)
    /// `startPoint` - a point under cursor for transform calculation, can be the picked point or else (world coordinates)
    /// Default implementation can be used as a reference for custom implementations
    MRVIEWER_API virtual TransformMode pick_( MouseButton button, int modifiers,
        std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint, Vector3f& startPoint );

    /// Helper function to calculate world bounding box for several objects
    /// Note: can be invalid (feature objects give an invalid box etc.)
    MRVIEWER_API Box3f getBbox_( const std::vector<std::shared_ptr<Object>>& objects );

private:
    int minDistance_ = 0;

    void clear_();

    void applyCurrentXf_( bool history );
    void resetXfs_();

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::vector<std::shared_ptr<Object>> objects_;
    std::vector<AffineXf3f> initialXfs_;

    TransformMode transformMode_ = TransformMode::None;
    Vector2i screenStartPoint_; // cNoPoint when moving actually started, {} when inactive
    AffineXf3f currentXf_;      // Transform currently applied to objects
    MouseButton currentButton_ = MouseButton::NoButton;

    // Data used to calculate transform
    Vector3f worldStartPoint_;  // World point corresponding to cursor, for transform calculation
    Vector3f xfCenterPoint_;
    float viewportStartPointZ_;
    Plane3f referencePlane_;
    float angle_ = 0.f;
    float shift_ = 0.f;
    float scale_ = 1.f;

    std::vector<ImVec2> visualizeVectors_;
};

}
