#pragma once
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPlane3.h"
#include "MRViewer/MRViewer.h"

namespace MR
{
// Visual widget to draw a plane
// present in scene (ancillary), subscribes to viewer events
class PlaneWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener>
{
    std::shared_ptr<ObjectMesh> planeObj_;
    Plane3f plane_;
    Box3f box_;
    Vector3f cameraUp3Old_;
    
    using OnPlaneUpdateCallback = std::function<void()>;
    OnPlaneUpdateCallback onPlaneUpdate_;

    bool pressed_ = false;
    Vector2f startMousePos_;
    Vector2f endMousePos_;

    std::shared_ptr<ObjectLines> line_;
public:
    
    // updates plane, triggers callback if it is specidied
    // if updateCameraRotation is true, plane transform will be updated with respect of camera up direction
    MRVIEWER_API void updatePlane( const Plane3f& plane, bool updateCameraRotation = true );
    // updates box which is used to calculate size of the plane
    // if updateCameraRotation is true, plane transform will be updated with respect of camera up direction
    MRVIEWER_API void updateBox( const Box3f& box, bool updateCameraRotation = true );
    // defines plane, adds plane object to scene
    MRVIEWER_API void definePlane();
    // undefines plane, removes PlaneObject from scene
    MRVIEWER_API void undefinePlane();

    // returns plane
    MRVIEWER_API const Plane3f& getPlane() const;
    // returns plane object
    MRVIEWER_API const std::shared_ptr<ObjectMesh>& getPlaneObject() const;
    // specifies callback onPlaneUpdate_
    // it is triggered when the method updatePlane is called
    MRVIEWER_API void setOnPlaneUpdateCalback( OnPlaneUpdateCallback  callback );

private:
    MRVIEWER_API void updateWidget_( bool updateCameraRotation = true );

    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
};
}
