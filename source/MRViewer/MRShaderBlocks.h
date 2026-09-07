#pragma once
#include "exports.h"
#include <string>

namespace MR
{

enum class ShaderTransparencyMode : char
{
    None,
    AlphaSort,
    DepthPeel
};

MRVIEWER_API std::string getPickerFragmentShader( bool points, bool cornerMode = true );

MRVIEWER_API std::string getFragmentShaderClippingBlock();

MRVIEWER_API std::string getFragmentShaderPointSizeBlock();

MRVIEWER_API std::string getFragmentShaderOnlyOddBlock( bool sampleMask );

MRVIEWER_API std::string getFragmentShaderHeaderBlock( bool gl4, bool alphaSort );

MRVIEWER_API std::string getFragmentShaderEndBlock( ShaderTransparencyMode transparencyMode );

MRVIEWER_API std::string getShaderMainBeginBlock( bool addDepthPeelSamplers );

}