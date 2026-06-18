// Single translation unit that emits the explicit instantiation of the openvdb::FloatGrid
// tree hierarchy. Every other MRVoxels TU declares it extern template via MRVoxelsPch.h and so
// skips regenerating the heavy OpenVDB tree codegen, linking to the copy emitted here instead.
#include "MRVoxels/MRVDBFloatGrid.h"

#define MR_VDB_INSTANTIATE template
#include "MRVoxels/MRVDBFloatGridInstantiations.inc"
#undef MR_VDB_INSTANTIATE
