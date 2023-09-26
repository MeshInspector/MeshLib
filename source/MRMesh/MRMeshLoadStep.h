#pragma once
#include "MRMeshFwd.h"

#include "MRExpected.h"
#include "MRProgressCallback.h"

#include <filesystem>

namespace MR::MeshLoad
{

MRMESH_API MR::Expected<MR::Mesh, std::string> fromStep( const std::filesystem::path& path, MR::Vector<MR::Color, MR::VertId>* colors = nullptr, MR::ProgressCallback callback = {} );
MRMESH_API MR::Expected<MR::Mesh, std::string> fromStep( std::istream& in, MR::Vector<MR::Color, MR::VertId>* colors = nullptr, MR::ProgressCallback callback = {} );

} // namespace MR::MeshLoad