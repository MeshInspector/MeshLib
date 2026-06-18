// Translation unit that builds MRVoxels's own precompiled header for the MSVC/MSBuild build
// (PrecompiledHeader=Create with MRVoxelsPch.h force-included). MRVoxels needs its own PCH
// (not the shared one) so it can additionally cache the heavy OpenVDB headers. CMake builds
// generate their own PCH-creating TU, so this file is excluded there (see CMakeLists.txt).
