#pragma once
#include "exports.h"
#include <string>

namespace MR
{

MRVIEWER_API std::string getLinesVertexShader();
MRVIEWER_API std::string getLinesFragmentShader( bool alphaSort );

MRVIEWER_API std::string getLinesJointVertexShader();
MRVIEWER_API std::string getLinesJointFragmentShader();

MRVIEWER_API std::string getLinesPickerVertexShader();
MRVIEWER_API std::string getLinesJointPickerVertexShader();

}