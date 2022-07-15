#pragma once
#include "MRMeshFwd.h"

namespace MR
{
//Creates a triangular prism. One edge of its base lies on 'x' axis and has 'baseLength' in length. 
//'leftAngle' and 'rightAngle' specify two adjacent angles
// axis of a prism is parallel to 'z' axis
    MRMESH_API Mesh makePrism( float baseLength, float leftAngle, float rightAngle, float height = 1.0f );
}