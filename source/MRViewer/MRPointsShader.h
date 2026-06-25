#pragma once
#include "exports.h"
#include "MRShaderBlocks.h"
#include <string>

namespace MR
{

MRVIEWER_API std::string getPointsVertexShader();
MRVIEWER_API std::string getPointsFragmentShader( ShaderTransparencyMode mode );

}