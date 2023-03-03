#include "MRVolumeShader.h"

#ifndef MR_GLSL_VERSION_LINE
#ifndef __EMSCRIPTEN__
#define MR_GLSL_VERSION_LINE R"(#version 150)"
#else
#define MR_GLSL_VERSION_LINE R"(#version 300 es)"
#endif
#endif

namespace MR
{

std::string getVolumeVertexQuadShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;
  in vec3 position;

  void main()
  {
    gl_Position = vec4 (position, 1.0);
  }
)";
}

std::string getVolumeFragmentShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;

  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;

  uniform vec4 viewport;
  uniform vec3 voxelSize;

  uniform sampler3D volume;
  uniform sampler2D denseMap;

  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane

  out vec4 outColor;

  void main()
  {
    mat4 fullM = proj * view * model;
    mat4 inverseFullM = inverse( fullM );

    vec3 clipNear = vec3( 2.0 * (gl_FragCoord.x - viewport.x ) / viewport.z - 1.0, 2.0 * (gl_FragCoord.y - viewport.y) / viewport.w - 1.0, -1.0 );
    vec3 clipFar = clipNear;
    clipFar.z = 1.0;

    vec4 rayStartW = inverseFullM * vec4( clipNear, 1.0 );
    vec4 rayEndW = inverseFullM * vec4( clipFar, 1.0 );

    vec3 rayStart = vec3( rayStartW.xyz ) / rayStartW.w;
    vec3 rayEnd = vec3( rayEndW.xyz ) / rayEndW.w;;
    vec3 normRayDir = normalize( rayEnd - rayStart );

    vec3 dims = vec3( textureSize( volume, 0 ) );
    vec3 minPoint = vec3( 0.0, 0.0, 0.0 );
    vec3 maxPoint = vec3( dims.x * voxelSize.x, dims.y * voxelSize.y, dims.z * voxelSize.z );

    // find planes intersection with ray
    vec3 minXInter = rayStart + ( ( minPoint.x - rayStart.x ) / normRayDir.x ) * normRayDir;
    vec3 maxXInter = rayStart + ( ( maxPoint.x - rayStart.x ) / normRayDir.x ) * normRayDir;

    vec3 minYInter = rayStart + ( ( minPoint.y - rayStart.y ) / normRayDir.y ) * normRayDir;
    vec3 maxYInter = rayStart + ( ( maxPoint.y - rayStart.y ) / normRayDir.y ) * normRayDir;

    vec3 minZInter = rayStart + ( ( minPoint.z - rayStart.z ) / normRayDir.z ) * normRayDir;
    vec3 maxZInter = rayStart + ( ( maxPoint.z - rayStart.z ) / normRayDir.z ) * normRayDir;

    // find min inter
    bool hasValidInter = false;
    float minValidInterDistSq = 0.0;
    vec3 minInter = vec3(0.0,0.0,0.0);

    if ( minXInter.y >= minPoint.y && minXInter.y <= maxPoint.y && minXInter.z >= minPoint.z && minXInter.z <= maxPoint.z )
    {
        hasValidInter = true;
        minValidInterDistSq = dot( minXInter - rayStart, minXInter - rayStart );
        minInter = minXInter;
    }

    if ( maxXInter.y >= minPoint.y && maxXInter.y <= maxPoint.y && maxXInter.z >= minPoint.z && maxXInter.z <= maxPoint.z )
    {
        float curDistSq = dot( maxXInter - rayStart, maxXInter - rayStart );
        if ( !hasValidInter || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInter = maxXInter;
            hasValidInter = true;
        }
    }

    if ( minYInter.x >= minPoint.x && minYInter.x <= maxPoint.x && minYInter.z >= minPoint.z && minYInter.z <= maxPoint.z )
    {
        float curDistSq = dot( minYInter - rayStart, minYInter - rayStart );
        if ( !hasValidInter || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInter = minYInter;
            hasValidInter = true;
        }
    }

    if ( maxYInter.x >= minPoint.x && maxYInter.x <= maxPoint.x && maxYInter.z >= minPoint.z && maxYInter.z <= maxPoint.z )
    {
        float curDistSq = dot( maxYInter - rayStart, maxYInter - rayStart );
        if ( !hasValidInter || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInter = maxYInter;
            hasValidInter = true;
        }
    }

    if ( minZInter.x >= minPoint.x && minZInter.x <= maxPoint.x && minZInter.y >= minPoint.y && minZInter.y <= maxPoint.y )
    {
        float curDistSq = dot( minZInter - rayStart, minZInter - rayStart );
        if ( !hasValidInter || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInter = minZInter;
            hasValidInter = true;
        }
    }

    if ( maxZInter.x >= minPoint.x && maxZInter.x <= maxPoint.x && maxZInter.y >= minPoint.y && maxZInter.y <= maxPoint.y )
    {
        float curDistSq = dot( maxZInter - rayStart, maxZInter - rayStart );
        if ( !hasValidInter || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInter = maxZInter;
            hasValidInter = true;
        }
    }

    if ( !hasValidInter )
        discard;

    // find ray step
    vec3 rayStep = vec3(0.0,0.0,0.0);
    if ( abs(normRayDir.x) >= abs(normRayDir.y) && abs(normRayDir.x) >= abs(normRayDir.z) )
        rayStep = ( voxelSize.x / abs(normRayDir.x) ) * normRayDir;
    else
    {
        if ( abs(normRayDir.y) >= abs(normRayDir.x) && abs(normRayDir.y) >= abs(normRayDir.z) )
            rayStep = ( voxelSize.y / abs(normRayDir.y) ) * normRayDir;
        else
            rayStep = ( voxelSize.z / abs(normRayDir.z) ) * normRayDir;
    }
    
    bool firstFound = false;
    vec3 firstOpaque = vec3(0.0,0.0,0.0);
    outColor = vec4(0.0,0.0,0.0,0.0);
    rayStart = minInter - rayStep * 0.5;
    while ( outColor.a < 1.0 )
    {
        rayStart = rayStart + rayStep;
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

        vec3 textCoord = vec3(0.0,0.0,0.0);
        textCoord.x = (rayStart.x - minPoint.x) / (maxPoint.x - minPoint.x);
        if ( textCoord.x < 0.0 || textCoord.x > 1.0 )
            break;

        textCoord.y = (rayStart.y - minPoint.y) / (maxPoint.y - minPoint.y);
        if ( textCoord.y < 0.0 || textCoord.y > 1.0 )
            break;

        textCoord.z = (rayStart.z - minPoint.z) / (maxPoint.z - minPoint.z);
        if ( textCoord.z < 0.0 || textCoord.z > 1.0 )
            break;
        
        float density = texture( volume, textCoord ).r;        
        vec4 color = texture( denseMap, vec2( density, 0.5 ) );
        float alpha = outColor.a + color.a * ( 1.0 - outColor.a );
        if ( alpha == 0.0 )
            continue;
        outColor.rgb = mix( color.a * color.rgb, outColor.rgb, outColor.a ) / alpha;
        outColor.a = alpha;
        if ( outColor.a > 0.98 )
            outColor.a = 1.0;

        if ( !firstFound )
        {
            firstFound = true;
            firstOpaque = rayStart;
        }
    }

    if ( outColor.a == 0.0 )
        discard;
    vec4 projCoord = fullM * vec4( firstOpaque, 1.0 );
    gl_FragDepth = projCoord.z / projCoord.w * 0.5 + 0.5;
  }
)";
}

}