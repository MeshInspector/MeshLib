#pragma once
#include <string>

namespace MR
{
// simple quad vertex shader
std::string getVolumeVertexQuadShader();

// shader with raytracing over volume
std::string getVolumeFragmentShader();

}