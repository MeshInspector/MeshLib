#pragma once
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPlane3.h"
#include "MRViewer/MRViewer.h"

namespace MR
{
struct PlaneVisualizer
{
    std::vector<std::shared_ptr<ObjectMeshHolder>> objects;

    std::shared_ptr<ObjectMesh> planeObj;
    Plane3f plane;
    Box3f objectsBox;

    AffineXf3f lastPlaneTransform;
    Vector3f cameraUp3Old;

    bool planeIsDefined = true;
    std::shared_ptr<ObjectLines> frameBorder;

    bool showPlane = true;
    bool pressed = false;

    ImVec2 startMousePos;
    ImVec2 endMousePos;

    bool importPlaneMode = false;
    bool clipByPlane = false;
    std::vector<boost::signals2::connection> xfChangedConnections;


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
