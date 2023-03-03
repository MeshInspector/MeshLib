#include "MRVolumeShader.h"
#include "MRGladGlfw.h"

namespace MR
{

std::string getTrivialVertexShader()
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
    vec3[6] intersections;
    intersections[0] = rayStart + ( ( minPoint.x - rayStart.x ) / normRayDir.x ) * normRayDir;
    intersections[1] = rayStart + ( ( maxPoint.x - rayStart.x ) / normRayDir.x ) * normRayDir;
    intersections[2] = rayStart + ( ( minPoint.y - rayStart.y ) / normRayDir.y ) * normRayDir;
    intersections[3] = rayStart + ( ( maxPoint.y - rayStart.y ) / normRayDir.y ) * normRayDir;
    intersections[4] = rayStart + ( ( minPoint.z - rayStart.z ) / normRayDir.z ) * normRayDir;
    intersections[5] = rayStart + ( ( maxPoint.z - rayStart.z ) / normRayDir.z ) * normRayDir;

    // find min inter
    float minValidInterDistSq = 0.0;
    int minInterIndex = -1;
    for ( int i = 0; i < 6; ++i )
    {
        vec2 comp;
        vec2 minComp;
        vec2 maxComp;
        if ( i < 2 )
        {
            comp = intersections[i].yz;
            minComp = minPoint.yz;
            maxComp = maxPoint.yz;
        }
        else if ( i < 4 )
        {
            comp = intersections[i].xz;
            minComp = minPoint.xz;
            maxComp = maxPoint.xz;
        }
        else
        {
            comp = intersections[i].xy;
            minComp = minPoint.xy;
            maxComp = maxPoint.xy;
        }
        if ( any( lessThan( comp, minComp ) ) || any( greaterThan( comp, maxComp ) ) )
            continue;
        float curDistSq = dot( intersections[i] - rayStart, intersections[i] - rayStart );
        if ( minInterIndex == -1 || curDistSq < minValidInterDistSq )
        {
            minValidInterDistSq = curDistSq;
            minInterIndex = i;
        }
    }

    if ( minInterIndex == -1 )
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
    rayStart = intersections[minInterIndex] - rayStep * 0.5;
    vec3 diagonal = maxPoint - minPoint;
    while ( outColor.a < 1.0 )
    {
        rayStart = rayStart + rayStep;
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

        vec3 textCoord = vec3(0.0,0.0,0.0);
        textCoord = ( rayStart - minPoint ) / diagonal;
        if ( any( lessThan( textCoord, vec3(0.0,0.0,0.0) ) ) || any( greaterThan( textCoord, vec3(1.0,1.0,1.0) ) ) )
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