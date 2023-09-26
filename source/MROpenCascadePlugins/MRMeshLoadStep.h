#pragma once
#include "exports.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"

#include <filesystem>

namespace MR::MeshLoad
{

MROPENCASCADEPLUGINS_API MR::Expected<MR::Mesh, std::string> fromStep( const std::filesystem::path& path, MR::Vector<MR::Color, MR::VertId>* colors = nullptr, MR::ProgressCallback callback = {} );
MROPENCASCADEPLUGINS_API MR::Expected<MR::Mesh, std::string> fromStep( std::istream& in, MR::Vector<MR::Color, MR::VertId>* colors = nullptr, MR::ProgressCallback callback = {} );

} // namespace MR::MeshLoad