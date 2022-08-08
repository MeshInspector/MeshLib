#include "MRMakePlane.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"

namespace MR
{

Mesh makePlane()
{
    Mesh plane;
    plane.points = {
        Vector3f{ -0.5f, -0.5f, 0.0f },
        Vector3f{ -0.5f,  0.5f, 0.0f },
        Vector3f{  0.5f,  0.5f, 0.0f },
        Vector3f{  0.5f, -0.5f, 0.0f }
    };

    Triangulation t{
        ThreeVertIds{ 2_v, 1_v, 0_v },
        ThreeVertIds{ 0_v, 3_v, 2_v }
    };
    plane.topology = MeshBuilder::fromTriangles( t );
    return plane;
}

}
