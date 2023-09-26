#pragma once
#ifdef _WIN32
#include "MRMeshFwd.h"

#include "MRExpected.h"
#include "MRProgressCallback.h"

#include <filesystem>

namespace MR::MeshLoad
{

/// loads meshes from STEP file using OpenCASCADE
MRMESH_API MR::Expected<MR::Mesh, std::string> fromStep( const std::filesystem::path& path, MR::VertColors* colors = nullptr, MR::ProgressCallback callback = {} );
MRMESH_API MR::Expected<MR::Mesh, std::string> fromStep( std::istream& in, MR::VertColors* colors = nullptr, MR::ProgressCallback callback = {} );

} // namespace MR::MeshLoad
#endif