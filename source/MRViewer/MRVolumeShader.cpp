#include "MRVolumeShader.h"
#include "MRGladGlfw.h"

namespace MR
{

std::string getTrivialVertexShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform sampler3D volume;
  uniform vec3 voxelSize;
  in vec3 position;

  void main()
  {
    vec3 dims = vec3( textureSize( volume, 0 ) );
    gl_Position = proj * view * model * vec4( voxelSize * dims * position, 1.0 );
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

    vec3 clipNear = vec3(0.0,0.0,0.0);
    clipNear.xy =  (2.0 * (gl_FragCoord.xy - viewport.xy)) / (viewport.zw) - vec2(1.0,1.0);
    clipNear.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / gl_DepthRange.diff;
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
    
    bool firstFound = false;
    vec3 firstOpaque = vec3(0.0,0.0,0.0);
    vec3 textCoord = vec3(0.0,0.0,0.0);
    outColor = vec4(0.0,0.0,0.0,0.0);
    vec3 rayStep = 0.5 * length( voxelSize ) * normRayDir;
    rayStart = rayStart - rayStep * 0.5;
    vec3 diagonal = maxPoint - minPoint;
    while ( outColor.a < 1.0 )
    {
        rayStart = rayStart + rayStep;

        textCoord = ( rayStart - minPoint ) / diagonal;
        if ( any( lessThan( textCoord, vec3(0.0,0.0,0.0) ) ) || any( greaterThan( textCoord, vec3(1.0,1.0,1.0) ) ) )
            break;
        
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

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

std::string getVolumePickerFragmentShader()
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

  uniform uint uniGeomId;
  out highp uvec4 outColor;

  void main()
  {
    mat4 fullM = proj * view * model;
    mat4 inverseFullM = inverse( fullM );

    vec3 clipNear = vec3(0.0,0.0,0.0);
    clipNear.xy =  (2.0 * (gl_FragCoord.xy - viewport.xy)) / (viewport.zw) - vec2(1.0,1.0);
    clipNear.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / gl_DepthRange.diff;
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
    
    bool firstFound = false;
    vec3 textCoord = vec3(0.0,0.0,0.0);
    vec3 rayStep = 0.5 * length( voxelSize ) * normRayDir;
    rayStart = rayStart - rayStep * 0.5;
    vec3 diagonal = maxPoint - minPoint;
    while ( !firstFound )
    {
        rayStart = rayStart + rayStep;

        textCoord = ( rayStart - minPoint ) / diagonal;
        if ( any( lessThan( textCoord, vec3(0.0,0.0,0.0) ) ) || any( greaterThan( textCoord, vec3(1.0,1.0,1.0) ) ) )
            break;
        
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

        float density = texture( volume, textCoord ).r;
        if ( texture( denseMap, vec2( density, 0.5 ) ).a == 0.0 )
            continue;
        firstFound = true;
    }

    if ( !firstFound )
        discard;
    vec4 projCoord = fullM * vec4( rayStart, 1.0 );
    float depth = projCoord.z / projCoord.w * 0.5 + 0.5;
    gl_FragDepth = depth;

    outColor.r = uint(0); // find VoxelId by world pos
    outColor.g = uniGeomId;

    outColor.a = uint(depth * 4294967295.0);
  }
)";
}

}