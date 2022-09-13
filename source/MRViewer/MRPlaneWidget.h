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
    
    std::function<void( const Plane3f& )> onPlaneUpdate_;

    bool pressed_ = false;
    ImVec2 startMousePos_;
    ImVec2 endMousePos_;

public:
    MRVIEWER_API PlaneWidget( const Plane3f& plane, const Box3f& box, std::function<void( const Plane3f& )> = nullptr );
    MRVIEWER_API ~PlaneWidget();
    
    MRVIEWER_API void updatePlane( const Plane3f& plane, bool updateCameraRotation = true );
    MRVIEWER_API void updateBox( const Box3f& box, bool updateCameraRotation = true );

private:
    MRVIEWER_API void updateWidget_( bool updateCameraRotation = true );

    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
};
}
