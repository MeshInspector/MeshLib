#include "MRShaderBlocks.h"
#include "MRGladGlfw.h"

namespace MR
{

std::string getPickerFragmentShader( bool points, bool cornerMode )
{
    const std::string head =
        MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;
  uniform bool useClippingPlane;
  uniform vec4 clippingPlane;
  uniform uint uniGeomId;

  in vec3 world_pos;
  
  in float primitiveIdf0;
  in float primitiveIdf1;

  out highp uvec4 color;
)";

    const std::string primId =
        cornerMode ? R"(
    uint primitiveId = ( uint(primitiveIdf1) << 20u ) + uint(primitiveIdf0);
)" : R"(
    uint primitiveId = uint(gl_PrimitiveID);
)";

    const std::string tail = R"(
    color.r = primitiveId;

    color.g = uniGeomId;

    color.a = uint(gl_FragCoord.z * 4294967295.0);
)";

    return
        head +
        getShaderMainBeginBlock( false ) +
        ( points ? getFragmentShaderPointSizeBlock() : R"()" ) +
        getFragmentShaderClippingBlock() +
        primId +
        tail +
        getFragmentShaderEndBlock( ShaderTransparencyMode::None );
}

std::string getFragmentShaderClippingBlock()
{
    return R"(
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;
)";
}

std::string getFragmentShaderPointSizeBlock()
{
    return R"(
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
)";
}

std::string getFragmentShaderOnlyOddBlock( bool sampleMask /*= false */ )
{
    if ( sampleMask )
        return R"(
    gl_SampleMask[0] = gl_SampleMaskIn[0];
    if (onlyOddFragments )
    {
      if (  mod( gl_FragCoord.x + gl_FragCoord.y, 2 ) < 1 )
        gl_SampleMask[0] = gl_SampleMaskIn[0] & 0xaaaaaaaa;
      else
        gl_SampleMask[0] = gl_SampleMaskIn[0] & 0x55555555;
    }
)";
    else
        return R"(
    if (onlyOddFragments && ((int(gl_FragCoord.x) + int(gl_FragCoord.y)) % 2) == 1)
      discard;
)";
}

std::string getFragmentShaderHeaderBlock( bool gl4, bool alphaSort )
{
    if ( !gl4 )
        return MR_GLSL_VERSION_LINE R"(
                    precision highp float;
            precision highp int;)";
    else if ( !alphaSort )
        return R"(#version 430 core)";
    else
        return R"(#version 430 core

  layout (early_fragment_tests) in;

  struct Node 
  {
    vec4 color;
    float depth;
    uint next;
  };

  layout (binding = 0, r32ui)      uniform uimage2D    heads;
  layout (binding = 0, offset = 0) uniform atomic_uint numNodes;

  layout (binding = 0, std430 ) buffer Lists
  {
    Node nodes [];
  };
)";
}

std::string getFragmentShaderEndBlock( ShaderTransparencyMode transparencyMode )
{
    if ( transparencyMode == ShaderTransparencyMode::AlphaSort )
        return R"(
    uint nodeIndex = atomicCounterIncrement ( numNodes );
    
    // is there any space ?
    {
        uint prev = imageAtomicExchange ( heads, ivec2 ( gl_FragCoord.xy ), nodeIndex );

        nodes [nodeIndex].color = outColor;
        nodes [nodeIndex].depth = gl_FragCoord.z;
        nodes [nodeIndex].next  = prev;
    }
    discard;
  }
)";
    else if ( transparencyMode == ShaderTransparencyMode::DepthPeel )
        return R"(

    ivec2 dpTexSize = textureSize( dp_bg_depths, 0 );
    ivec2 dpTexCoord = clamp( ivec2(gl_FragCoord.xy), ivec2(0,0), dpTexSize-ivec2(1,1));
    
    float dpBGDepth = texelFetch( dp_bg_depths, dpTexCoord,0 ).r;
    float dpFGDepth = texelFetch( dp_fg_depths, dpTexCoord,0 ).r;
    vec4 dpColor = texelFetch( dp_fg_colors, dpTexCoord,0 );

    float dp_eps = 0.00001;
    if ( gl_FragCoord.z >= dpBGDepth )
    {
        discard;
    }
    else if ( gl_FragCoord.z <= dpFGDepth )
    {
        outColor = dpColor;
        gl_FragDepth = dpBGDepth - dp_eps;
    }
    else 
    {
        // blend
        float alpha = dpColor.a + outColor.a * ( 1.0 - dpColor.a );
        outColor.rgb = mix( outColor.a * outColor.rgb, dpColor.rgb, dpColor.a ) / alpha;
        outColor.a = alpha;
        gl_FragDepth = gl_FragCoord.z;
    }
  }
)";
    else
        return R"(
  }
)";
}

std::string getShaderMainBeginBlock( bool addDepthPeelSamplers )
{
    if (!addDepthPeelSamplers )
        return R"(
  void main()
  {
)";
    else
        return R"(
  uniform sampler2D dp_bg_depths;
  uniform sampler2D dp_fg_colors;
  uniform sampler2D dp_fg_depths;

  void main()
  {
)";
}

}