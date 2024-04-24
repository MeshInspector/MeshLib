#pragma once
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRViewer/MRViewerInstance.h"
#include "imgui.h"

namespace MR
{

class Object;
class Viewer;

class MRVIEWER_CLASS MoveObjectByMouseImpl
{
public:
    MoveObjectByMouseImpl() = default;
    virtual ~MoveObjectByMouseImpl() = default;

    // Drawing callback to draw lines and tooltips
    // Should be called from drawDialog
    MRVIEWER_API void onDrawDialog( float menuScaling );

    // These functions should be called from corresponding mouse handlers
    // Return true if handled (picked/moved/released)
    MRVIEWER_API bool onMouseDown( MouseButton button, int modifiers );
    MRVIEWER_API bool onMouseMove( int x, int y );
    MRVIEWER_API bool onMouseUp( MouseButton button, int modifiers );

    // Current object being moved; null if no object is being moved right now
    std::shared_ptr<VisualObject> currentObject() { return obj_; }

    // Reset transformation and stop moving the object. Does nothing if not moving anything
    // Calling onMouseUp is not necessary after this
    // Should be called when closing plugin etc.
    MRVIEWER_API void cancel();

protected:
    // Called when onMouseDown_ picks an object
    // Return true to start transformation, false to skip
    // `obj` and `point` can be modified if necessary (in `point`, only `point` member is used)
    // Default implementation returns true for all non-ancillary objects;
    MRVIEWER_API virtual bool onPick( 
        std::shared_ptr<VisualObject>& obj, PointOnObject& point, int modifiers );

    // Returns a list of objects that will be moved
    // Default implementation returns `{ obj }`
    // Regardless of the list, `obj` is used as a reference object in transformation
    MRVIEWER_API virtual std::vector<std::shared_ptr<Object>> getObjects(
        const std::shared_ptr<VisualObject>& obj, const PointOnObject& point, int modifiers );

private:
    Viewer* viewer = &getViewerInstance();

    void clear_();

    void setWorldXf_( AffineXf3f worldXf, bool history );
    void resetWorldXf_();

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::shared_ptr<VisualObject> obj_;

    std::vector<std::shared_ptr<Object>> objects_;
    std::vector<AffineXf3f> objectsXfs_;

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
