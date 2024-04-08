#pragma once
#include "MRMeshFwd.h"
#include "MRConstants.h"

namespace MR
{
// Z-looking

    //Draws cylinder with radius 'radius', height - 'length', its base have 'resolution' sides
    MRMESH_API Mesh makeCylinder( float radius = 0.1f, float length = 1.0f, int resolution = 16 );

    // A hollow cylinder.
    MRMESH_API Mesh makeOpenCylinder( float radius = 1, float z1 = -1, float z2 = 1, int numCircleSegments = 16 );

    // A hollow cone.
    MRMESH_API Mesh makeOpenCone( float radius = 1, float zApex = 0, float zBase = 1, int numCircleSegments = 16 );

    MRMESH_API Mesh makeCylinderAdvanced( float radius0 = 0.1f, float radius1 = 0.1f, float start_angle = 0.0f, float arc_size = 2.0f * PI_F, float length = 1.0f, int resolution = 16 );

    // Makes cone mesh by calling makeCylinderAdvanced with the top radius 0.
    MRMESH_API Mesh makeCone( float radius0 = 0.1f, float length = 1.0f, int resolution = 32 );
}
