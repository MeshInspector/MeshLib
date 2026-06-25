#pragma once
#include "exports.h"
#include "MRShaderBlocks.h"
#include <string>

namespace MR
{

MRVIEWER_API std::string getMeshVerticesShader();

MRVIEWER_API std::string getMeshFragmentShader( bool gl4, ShaderTransparencyMode mode, bool msaaEnabled );

MRVIEWER_API std::string getMeshFragmentShaderArgumetsBlock();

MRVIEWER_API std::string getMeshFragmentShaderColoringBlock();

}