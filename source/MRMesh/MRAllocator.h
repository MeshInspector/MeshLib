#pragma once

#include "MRMeshFwd.h"
#include <memory>

// on Linux, MR_MIMALLOC_OVERRIDE executables forward malloc to the same mimalloc, see mr_enable_mimalloc
#if !defined( __EMSCRIPTEN__ ) && ( !defined( MR_MIMALLOC_OVERRIDE ) || defined( __linux__ ) )
#define MR_ALLOCATOR_IS_MIMALLOC
#endif

#ifdef MR_ALLOCATOR_IS_MIMALLOC
#include <mimalloc.h>
#if defined( _MSC_VER ) && !defined( MR_USE_CMAKE_CONFIGURE_FILE )
// the inline mi_stl_allocator calls mimalloc from every binary using MRMesh; MeshLib's CMake targets link it instead
#ifdef _DEBUG
#pragma comment( lib, "mimalloc-debug.dll.lib" )
#else
#pragma comment( lib, "mimalloc.dll.lib" )
#endif
#endif
#endif

namespace MR
{

/// allocator of MeshLib containers that benefit from mimalloc: mimalloc explicitly in MeshLib's own builds,
/// std::allocator where malloc is mimalloc already (Emscripten's -sMALLOC=mimalloc, MR_MIMALLOC_OVERRIDE executables on Windows and macOS)
#ifdef MR_ALLOCATOR_IS_MIMALLOC
template <typename T>
using Allocator = mi_stl_allocator<T>;
#else
template <typename T>
using Allocator = std::allocator<T>;
#endif

} // namespace MR
