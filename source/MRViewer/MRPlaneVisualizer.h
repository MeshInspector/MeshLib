#pragma once
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPlane3.h"
#include "MRViewer/MRViewer.h"

namespace MR
{
struct PlaneVisualizer
{
    std::vector<std::shared_ptr<ObjectMeshHolder>> objects_;

    std::shared_ptr<ObjectMesh> planeObj_;
    Plane3f plane_;
    Box3f objectsBox_;

    AffineXf3f lastPlaneTransform_;
    Vector3f cameraUp3Old_;

    bool planeIsDefined_ = true;
    std::shared_ptr<ObjectLines> frameBorder_;

    bool showPlane_ = true;
    bool pressed_ = false;

    ImVec2 startMousePos_;
    ImVec2 endMousePos_;

    bool importPlaneMode_{ false };
    std::vector<boost::signals2::connection> xfChangedConnections_;  


    MRVIEWER_API PlaneVisualizer();
    MRVIEWER_API ~PlaneVisualizer();

    MRVIEWER_API void setupPlane();
    MRVIEWER_API void updatePlane( bool updateCameraRotation = true );
    MRVIEWER_API void updateXfs();
    MRVIEWER_API void definePlane();
    MRVIEWER_API void undefinePlane();
    MRVIEWER_API void setupFrameBorder();
    MRVIEWER_API void updateFrameBorder();
};
}
