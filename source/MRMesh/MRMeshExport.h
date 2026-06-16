// This header is force-included (after the precompiled MRPch.h) into every MRMesh
// translation unit on Windows; it is not meant to be #included manually.
//
// When MR_PCH_USE_EXTRA_HEADERS is set, the shared precompiled header pulls in
// MRMeshFwd.h. That precompiled header is produced by the MRPch project, which does
// not define MRMesh_EXPORTS, so MRMESH_API is baked as __declspec(dllimport).
// MRMesh itself must export those symbols, so restore the macro to dllexport here.
#ifdef _WIN32
#undef MRMESH_API
#define MRMESH_API __declspec(dllexport)
#endif