#pragma once
// Own precompiled header for MRVoxels (MSVC/MSBuild + CMake-with-MSVC only). On top of the
// shared MRPch.h content it caches the OpenVDB stack, which dominates front-end parsing in
// the voxels TUs. MRPch.h is included first so MRTBB.h suppresses TBB's warnings before
// OpenVDB pulls TBB in. MESHLIB_NO_VIEWER (set by the build) drops MRPch.h's viewer/imgui
// section, which MRVoxels (a lower layer) neither needs nor has on its include path.
#include "MRPch/MRPch.h"
#include "MRVoxels/MRVDBFloatGrid.h"

// Home the heavy OpenVDB FloatGrid tree codegen in one TU (MRVDBFloatGridInstantiation.cpp):
// declaring the hierarchy extern template here means every voxels TU using this PCH skips
// regenerating it. (The vcpkg OpenVDB is built without explicit instantiation, so we emit it.)
#define MR_VDB_INSTANTIATE extern template
#include "MRVoxels/MRVDBFloatGridInstantiations.inc"
#undef MR_VDB_INSTANTIATE
