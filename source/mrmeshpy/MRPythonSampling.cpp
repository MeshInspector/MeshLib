#include "MRMesh/MRPython.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRGridSampling.h"
#include "MRMesh/MRUniformSampling.h"

MR_ADD_PYTHON_FUNCTION( mrmeshpy, sample_points_by_grid, MR::pointGridSampling, "simplifies PointCloud by sampling points using voxels" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, sample_points_uniform, MR::pointUniformSampling, "simplifies PointCloud by sampling points using distance between them" )
