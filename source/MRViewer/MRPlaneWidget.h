#pragma once
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPlane3.h"
#include "MRViewer/MRViewer.h"

namespace MR
{
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

public:
    MRVIEWER_API PlaneWidget( const Plane3f& plane, const Box3f& box, OnPlaneUpdateCallback = nullptr );
    MRVIEWER_API ~PlaneWidget();
    
    MRVIEWER_API void updatePlane( const Plane3f& plane, bool updateCameraRotation = true );
    MRVIEWER_API void updateBox( const Box3f& box, bool updateCameraRotation = true );

    MRVIEWER_API void definePlane();

    MRVIEWER_API void undefinePlane();

    MRVIEWER_API const Plane3f& getPlane() const;
    MRVIEWER_API const std::shared_ptr<ObjectMesh>& getPlaneObject() const;
    MRVIEWER_API void setOnPlaneUpdateCalback( OnPlaneUpdateCallback  callback );

private:
    MRVIEWER_API void updateWidget_( bool updateCameraRotation = true );

    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
};
}
