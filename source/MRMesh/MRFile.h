#pragma once

#include "MRMeshFwd.h"
#include <filesystem>
#include <cstdio>

namespace MR
{

// this version of fopen unlike std::fopen supports unicode file names on Windows
MRMESH_API FILE * fopen( const std::filesystem::path & filename, const char * mode );

} //namespace MR
