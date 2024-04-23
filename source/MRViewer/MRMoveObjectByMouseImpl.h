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
    MRVIEWER_API bool onMouseDown( MouseButton button, int modifier );
    MRVIEWER_API bool onMouseMove( int x, int y );
    MRVIEWER_API bool onMouseUp( MouseButton btn, int modifiers );

    // Current object being moved; null if no object is being moved right now
    std::shared_ptr<VisualObject> currentObject() { return obj_; }

    // Reset transformation and stop moving the object. Does nothing if not moving anything
    // Calling onMouseUp is not necessary after this
    MRVIEWER_API void cancel();

protected:
    // Called when onMouseDown_ picks an object
    // Return true to start transformation, false to skip; modify the arguments if necessary
    // Default implementation returns true for all non-ancillary objects
    MRVIEWER_API virtual bool onPick( std::shared_ptr<VisualObject> &obj, PointOnObject &point );

private:
    Viewer* viewer = &getViewerInstance();

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::shared_ptr<VisualObject> obj_;

    Vector3f worldStartPoint_;
    Vector3f worldBboxCenter_;
    Vector3f bboxCenter_;
    AffineXf3f objWorldXf_;
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
