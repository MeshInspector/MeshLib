#pragma once
#include <string>

namespace MR
{
// simple quad vertex shader
std::string getTrivialVertexShader();

// shader with raytracing over volume
std::string getVolumeFragmentShader();

// shader with raytracing over volume
std::string getVolumePickerFragmentShader();

}