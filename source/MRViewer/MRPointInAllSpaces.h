#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRViewportId.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRPointOnFace.h>
#include <memory>

namespace MR
{

struct PointInAllSpaces
{
    // Screen space: window x[0, window.width] y[0, window.height]. [0,0] top-left corner of the window
    // Z [0,1] - 0 is Dnear, 1 is Dfar
    Vector3f screenSpace;

    // Viewport space: viewport x[0, viewport.width] y[0, viewport.height]. [0,0] top-left corner of the viewport with that Id
    // Z [0,1] - 0 is Dnear, 1 is Dfar
    Vector3f viewportSpace;
    ViewportId viewportId;

    // Clip space: viewport xyz[-1.f, 1.f]. [0, 0, -1] middle point of the viewport on Dnear. [0, 0, 1] middle point of the viewport on Dfar.
    // [-1, -1, -1] is lower left Dnear corner; [1, 1, 1] is upper right Dfar corner
    Vector3f clipSpace;

    // Camera space: applied view affine transform to world points xyz[-inf, inf]. [0, 0, 0] middle point of the viewport on Dnear.
    // X axis goes on the right. Y axis goes up. Z axis goes backward.
    Vector3f cameraSpace;

    // World space: applied model transform to Mesh(Point Cloud) vertices xyz[-inf, inf].
    Vector3f worldSpace;

    // Model space: coordinates as they stored in the model of VisualObject
    std::shared_ptr<VisualObject> obj;
    PointOnFace pof;
};

} //namespace MR
