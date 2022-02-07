#include "MRMesh/MRPython.h"
#include "MRMesh/MRMeshToPointCloud.h"
#include "MRMesh/MRMesh.h"

MR_ADD_PYTHON_FUNCTION( mrmeshpy, mesh_to_points, &MR::meshToPointCloud, "bool saveNormals: calculate from triangles and save normal for each point" )
