#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh\MRMeshDistance.h"

using namespace MR;

MR_ADD_PYTHON_FUNCTION( mrmeshpy, findMaxMeshDistance, &findMaxDistance, "returns the maximum of the distances from each mesh point to another mesh" )