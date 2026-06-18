// Single TU that emits the explicit instantiation of the openvdb::FloatGrid tree hierarchy
// (exported from MRVoxels). Defining MR_VOXELS_INSTANTIATE_FLOATGRID makes MROpenVDB.h emit the
// definitions instead of extern-template declarations; every other TU links to the copy here.
#define MR_VOXELS_INSTANTIATE_FLOATGRID
#include <MRVoxels/MROpenVDB.h>
