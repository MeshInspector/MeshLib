// Translation unit that builds MRViewer's own precompiled header for the MSVC/MSBuild build
// (PrecompiledHeader=Create with MRPch.h force-included). MRViewer cannot reuse the shared
// MRPch.pch because MRVIEWER_API is dllexport here but dllimport in that PCH. CMake builds
// generate their own PCH-creating TU, so this file is excluded there (see CMakeLists.txt).
