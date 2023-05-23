#include "MRLinesShader.h"
#include "MRShaderBlocks.h"
#include "MRGladGlfw.h"

#ifndef __EMSCRIPTEN__
#define MR_GLSL_VERSION_LINE_330 R"(#version 330)"
#else
#define MR_GLSL_VERSION_LINE_330 MR_GLSL_VERSION_LINE
#endif

namespace
{
std::string getLinesVertexShaderBaseArgumentsBlock( bool points )
{
    std::string base = R"(
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform highp usampler2D vertices;
  out vec3 world_pos;    // (out to fragment shader) vert transformed position
  out float primitiveIdf0;
  out float primitiveIdf1;
)";
    if ( points )
        base += R"(
  uniform float pointSize;
)";
    return base;
}

std::string getLinesVertexShaderColorsArgumentsBlock()
{
    return R"(
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform sampler2D vertColors;
  out vec4 Ki;           // (out to fragment shader) vert color 
)";
}

std::string getLinesVertexShaderWidthArgumentsBlock()
{
    return R"(
  uniform vec4 viewport;
  uniform float width;
)";
}

std::string getLinesShaderHeaderBlock()
{
    return MR_GLSL_VERSION_LINE_330 R"(
            precision highp float;
            precision highp int;
)";
}

std::string getLinesFragmentShaderArgumentsBlock()
{
    return R"(
  uniform sampler2D lineColors;  // (in from base) line color
  uniform bool perLineColoring;      // (in from base) use lines colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
 
  uniform vec4 mainColor;            // (in from base) color if colormap is off
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane  

  in float primitiveIdf0;
  in float primitiveIdf1;

  uniform float globalAlpha;        // (in from base) global transparency multiplier

  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
                                     
  out vec4 outColor;                 // (out to render) fragment color
)";
}

std::string getLinesFragmentShaderColoringBlock()
{
    return R"(
    uint primitiveId = ( uint(primitiveIdf1) << 20u ) + uint(primitiveIdf0);
    vec4 colorCpy = mainColor;
    if ( perVertColoring )
    {
      colorCpy = Ki;      
    }
    if ( perLineColoring )
    {
      ivec2 texSize = textureSize( lineColors, 0 );
      colorCpy = texelFetch(lineColors, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 );
    }
    outColor = vec4(colorCpy.rgb,colorCpy.a * globalAlpha);
    if (outColor.a == 0.0)
      discard;
)";
}

std::string getLinesVertexShaderPositionBlock()
{
    return R"(
    uint baseLineId = uint(gl_VertexID) / 6u;
    uint interVertId = uint(gl_VertexID) % 6u;
    bool top = interVertId > 3u || interVertId == 2u;
    bool left = interVertId != 1u && interVertId != 5u;
    uint baseCoordId = 2u * baseLineId + uint( top );
    uint otherCoordId = baseCoordId ^ 1u;

    ivec2 vTexSize = textureSize( vertices, 0 );
    uvec3 uBasePos = texelFetch( vertices, ivec2( baseCoordId % uint(vTexSize.x), baseCoordId / uint(vTexSize.x) ), 0 ).rgb;
    uvec3 uOtherPos = texelFetch( vertices, ivec2( otherCoordId % uint(vTexSize.x), otherCoordId / uint(vTexSize.x) ), 0 ).rgb;
    
    vec3 basePos = uintBitsToFloat( uBasePos );
    vec3 otherPos = uintBitsToFloat( uOtherPos );

    world_pos = vec3( model * vec4 ( basePos, 1.0 ) );
    vec3 otherWorldPos = vec3( model * vec4 ( otherPos, 1.0 ) );
    
    vec3 posEye = vec3( view * vec4( world_pos, 1.0 ) );
    vec3 otherPosEye = vec3( view * vec4( otherWorldPos, 1.0 ) );

    vec4 projBasePos = proj * vec4( posEye, 1.0 );
    vec4 projOtherPos = proj * vec4( otherPosEye, 1.0 );
    vec2 basePix = viewport.xy + viewport.zw * ( vec2(1.0,1.0) + projBasePos.xy / projBasePos.w ) * 0.5;
    vec2 otherPix = viewport.xy + viewport.zw * ( vec2(1.0,1.0) + projOtherPos.xy / projOtherPos.w ) * 0.5;

    vec2 dir = normalize( otherPix - basePix );
    dir.xy = dir.yx;
    if ( left )
        dir.x = -dir.x;
    else
        dir.y = -dir.y;

    basePix = basePix + dir * 0.5 * width;

    projBasePos.xy = ( 2.0 * (basePix - viewport.xy) / (viewport.zw) - vec2(1.0,1.0) ) * projBasePos.w;
    
    gl_Position = projBasePos;

    primitiveIdf1 = float( uint( baseLineId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( baseLineId % uint( 1u << 20u ) ) + 0.5;
)";
}

std::string getLinesJointVertexShaderPositionBlock()
{
    return R"(
    uint baseLineId = uint(gl_VertexID) / 2u;
    uint interVertId = uint(gl_VertexID) % 2u;
    uint baseCoordId = 2u * baseLineId + interVertId;

    ivec2 vTexSize = textureSize( vertices, 0 );
    uvec3 uBasePos = texelFetch( vertices, ivec2( baseCoordId % uint(vTexSize.x), baseCoordId / uint(vTexSize.x) ), 0 ).rgb;    
    vec3 basePos = uintBitsToFloat( uBasePos );

    world_pos = vec3( model * vec4( basePos, 1.0 ) );
    gl_Position = proj * view * vec4( world_pos, 1.0 );

    primitiveIdf1 = float( uint( baseLineId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( baseLineId % uint( 1u << 20u ) ) + 0.5;
    gl_PointSize = pointSize;
)";
}

std::string getLinesVertexShaderColoringBlock()
{
    return R"(
    Ki = vec4(0.0);
    if ( perVertColoring )
    {  
        ivec2 vcTexSize = textureSize( vertColors, 0 );
        Ki = texelFetch( vertColors, ivec2( baseCoordId % uint(vcTexSize.x), baseCoordId / uint(vcTexSize.x) ), 0 );
    }
)";
}
}

namespace MR
{

std::string getLinesVertexShader()
{
    return
        getLinesShaderHeaderBlock() +
        getLinesVertexShaderBaseArgumentsBlock( false ) +
        getLinesVertexShaderColorsArgumentsBlock() +
        getLinesVertexShaderWidthArgumentsBlock() +
        getShaderMainBeginBlock() +
        getLinesVertexShaderPositionBlock() +
        getLinesVertexShaderColoringBlock() +
        getFragmentShaderEndBlock( false );
}

std::string getLinesFragmentShader( bool alphaSort )
{
    return
        getFragmentShaderHeaderBlock( alphaSort, alphaSort ) +
        getLinesFragmentShaderArgumentsBlock() +
        getShaderMainBeginBlock() +
        getFragmentShaderClippingBlock() +
        getLinesFragmentShaderColoringBlock() +
        getFragmentShaderEndBlock( alphaSort );
}

std::string getLinesJointVertexShader()
{
    return 
        getLinesShaderHeaderBlock() +
        getLinesVertexShaderBaseArgumentsBlock( true ) +
        getLinesVertexShaderColorsArgumentsBlock() +
        getShaderMainBeginBlock() +
        getLinesJointVertexShaderPositionBlock() +
        getLinesVertexShaderColoringBlock() +
        getFragmentShaderEndBlock( false );
}

std::string getLinesJointFragmentShader()
{
    return
        getLinesShaderHeaderBlock() +
        getLinesFragmentShaderArgumentsBlock() +
        getShaderMainBeginBlock() +
        getFragmentShaderPointSizeBlock() +
        getFragmentShaderClippingBlock() +
        getLinesFragmentShaderColoringBlock() +
        getFragmentShaderEndBlock( false );
}

std::string getLinesPickerVertexShader()
{
    return
        getLinesShaderHeaderBlock() +
        getLinesVertexShaderBaseArgumentsBlock( false ) +
        getLinesVertexShaderWidthArgumentsBlock() +
        getShaderMainBeginBlock() +
        getLinesVertexShaderPositionBlock() +
        getFragmentShaderEndBlock( false );
}

std::string getLinesJointPickerVertexShader()
{
    return
        getLinesShaderHeaderBlock() +
        getLinesVertexShaderBaseArgumentsBlock( true ) +
        getShaderMainBeginBlock() +
        getLinesJointVertexShaderPositionBlock() +
        getFragmentShaderEndBlock( false );
}

}