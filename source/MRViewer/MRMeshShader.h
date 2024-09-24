#pragma once
#include "exports.h"
#include <string>

namespace MR
{

MRVIEWER_API std::string getMeshVerticesShader();

MRVIEWER_API std::string getMeshFragmentShader( bool gl4, bool alphaSort, bool msaaEnabled );

MRVIEWER_API std::string getMeshFragmentShaderArgumetsBlock();

MRVIEWER_API std::string getMeshFragmentShaderColoringBlock();

}