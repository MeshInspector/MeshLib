#pragma once
#include "exports.h"
#include <string>

namespace MR
{
// simple quad vertex shader
MRVIEWER_API std::string getTrivialVertexShader();

// shader with raytracing over volume
MRVIEWER_API std::string getVolumeFragmentShader();

// shader with raytracing over volume
MRVIEWER_API std::string getVolumePickerFragmentShader();

}