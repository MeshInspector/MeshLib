#pragma once
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRViewer/MRViewerFwd.h"
#include "imgui.h"

namespace MR
{

/// Helper class to incorporate basic object transformation feature into plugins
/// User can move objects by dragging them, rotate by dragging with Ctrl key
/// To use, create class instance and call its event handlers
/// For extra features, override the pick and object selection functions
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

    /// These functions should be called from corresponding mouse handlers (or better drag handlers)
    /// Return true if handled (picked/moved/released)
    /// It is recommended to call `viewer->select_hovered_viewport()` before `onMouseDown`
    /// The object is transformed temporarily in `onMouseMove`
    /// `onMouseUp` finalizes the transformation and writes to undo/redo history (unless cancelled)
    MRVIEWER_API bool onMouseDown( MouseButton button, int modifiers );
    MRVIEWER_API bool onMouseMove( int x, int y );
    MRVIEWER_API bool onMouseUp( MouseButton button, int modifiers );

    /// Object currently picked for moving (if any, otherwise null)
    /// The current object is set in `onMouseDown` and set to null in `onMouseUp`
    std::shared_ptr<VisualObject> currentObject() const { return obj_; }

    /// Returns true if has a picked object, and actually started moving it
    /// Return false if no object, or `minDistance` has not yet reached
    MRVIEWER_API bool isMoving() const;

    /// Reset transformation and stop moving the object(s). Does nothing if not moving anything
    /// Calling `onMouseUp` is not necessary after this
    /// Should be called when closing plugin etc.
    MRVIEWER_API void cancel();

protected:
    /// Called when `onMouseDown` picks an object
    /// Return true to start transformation, false to skip
    /// `obj` and `point` can be modified if necessary (in `point`, only `point` member is used)
    /// Default implementation returns true for all non-ancillary objects
    MRVIEWER_API virtual bool onPick_(
        std::shared_ptr<VisualObject>& obj, PointOnObject& point, int modifiers );

    // Returns a list of objects that will be moved
    // Default implementation returns `{ obj }`
    // Regardless of the list, `obj` is used as a reference object in transformation
    MRVIEWER_API virtual std::vector<std::shared_ptr<Object>> getObjects_(
        const std::shared_ptr<VisualObject>& obj, const PointOnObject& point, int modifiers );

private:
    int minDistance_ = 0;

    void clear_();

    void setWorldXf_( AffineXf3f worldXf, bool history );
    void resetWorldXf_();

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::shared_ptr<VisualObject> obj_;

    std::vector<std::shared_ptr<Object>> objects_;
    std::vector<AffineXf3f> objectsXfs_;

    Vector2i screenStartPoint_; // cNoPoint when moving actually started, {} when inactive
    Vector3f worldStartPoint_;
    Vector3f worldBboxCenter_;
    Vector3f bboxCenter_;
    AffineXf3f objWorldXf_;
    AffineXf3f newWorldXf_;
    float viewportStartPointZ_;
    Plane3f rotationPlane_;

    std::vector<ImVec2> visualizeVectors_;
    float angle_ = 0.f;
    float shift_ = 0.f;

    enum class TransformMode
    {
        Translation,
        Rotation,
        None
    } transformMode_ = TransformMode::None;
};

}
