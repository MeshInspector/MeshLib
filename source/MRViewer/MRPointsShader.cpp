#include "MRPointsShader.h"
#include "MRShaderBlocks.h"
#include "MRGladGlfw.h"

namespace MR
{

// can use in Mesh Vertex & Fragment shaders
std::string getPointsShaderHeaderBlock()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;
)";
}

// can use in Mesh Vertex & Fragment shaders
std::string getPointsShaderViewBlock()
{
    return R"(
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;
)";
}

std::string getPointsVertexSpecificGlobalArgumentBlock()
{
    return R"(
  uniform float pointSize;
  uniform uint primBucketSize;
)";
}

// can use in Mesh Vertex shaders
std::string getPointsVertexInputArgumentBlock()
{
    return R"(
  in vec3 position;       // (in from base) vert position
  in vec3 normal;         // (in from base) vert normal
  in vec4 K;              // (in from base) vert color
)";
}

// can use in Mesh Vertex shaders
std::string getPointsVertexOutputArgumentBlock()
{
    return R"(
  out vec3 world_pos;    // (out to fragment shader) vert transformed position
  out vec4 Ki;           // (out to fragment shader) vert color 
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)
  out float primitiveIdf0;
  out float primitiveIdf1;
)";
}

// can use in Mesh Vertex shaders
std::string getPointsVertexMainBeginBlock()
{
    return
        getShaderMainBeginBlock() +
        R"(
    world_pos = vec3(model*vec4 (position, 1.0));
    position_eye = vec3 (view * vec4 (world_pos, 1.0));
    normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
    normal_eye = normalize(normal_eye);
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    Ki = K;
)";
}

std::string getPointsVertexMainSpecificBlock()
{
    return R"(
    gl_PointSize = pointSize;
    uint primId = uint(gl_VertexID) / primBucketSize;
)";
}

// can use in Mesh Vertex shaders
std::string getPointsVertexMainEndBlock()
{
    return R"(
    primitiveIdf1 = float( uint( primId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( primId % uint( 1u << 20u ) ) + 0.5;
  }
)";
}

///////////////////////////////////////

std::string getPointsFragmentShaderArgumetsBlock()
{
    return R"(
  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform bool showSelVerts;    // (in from base) use selection or not
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 selBackColor;   // (in from base) selection back face color

  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool hasNormals;           // (in from base) dont use normals if they are not
 
  uniform vec4 mainColor;            // (in from base) color if colormap is off
  uniform vec4 backColor;            // (in from base) back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true

  uniform float specExp;   // (in from base) lighting parameter
  uniform vec3 ligthPosEye;   // (in from base) light position transformed by view only (not proj)
                                     
  uniform float ambientStrength;    // (in from base) non-directional lighting
  uniform float specularStrength;   // (in from base) reflection intensity
  uniform float globalAlpha;        // (in from base) global transparency multiplier
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position

  in float primitiveIdf0;
  in float primitiveIdf1;
                                     
  out vec4 outColor;                 // (out to render) fragment color
)";
}

std::string getPointsFragmentShaderColoringBlock()
{
    return R"(
    vec3 normEyeCpy = normal_eye;
    
    vec3 vector_to_light_eye = ligthPosEye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    if (!hasNormals)
      normEyeCpy = direction_to_light_eye;

    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
    
    uint primitiveId = ( uint(primitiveIdf1) << 20u ) + uint(primitiveIdf0);
    vec4 colorCpy;
    bool selected = false;
    if ( showSelVerts )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }

    bool frontFacing = dot_prod >= 0.0;
    if ( frontFacing == invertNormals )
    {
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selBackColor;
    }
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if (!frontFacing)
      dot_prod = -dot_prod;
    if ( dot_prod < 0.0 )
      dot_prod = 0.0;

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if ( dot_prod_specular < 0.0 )
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specExp);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a * globalAlpha);
    if (outColor.a == 0.0)
      discard;
)";
}

}

/////////////////////////////////////////////////////////////////////////////

namespace MR
{

std::string getPointsVertexShader()
{
    return
        getPointsShaderHeaderBlock() +
        getPointsShaderViewBlock() +
        getPointsVertexSpecificGlobalArgumentBlock() +
        getPointsVertexInputArgumentBlock() +
        getPointsVertexOutputArgumentBlock() +
        getPointsVertexMainBeginBlock() +
        getPointsVertexMainSpecificBlock() +
        getPointsVertexMainEndBlock();
}

std::string getPointsFragmentShader( bool alphaSort )
{
    return
        getFragmentShaderHeaderBlock( alphaSort, alphaSort ) +
        getPointsShaderViewBlock() +
        getPointsFragmentShaderArgumetsBlock() +
        getShaderMainBeginBlock() +
        getFragmentShaderPointSizeBlock() +
        getFragmentShaderClippingBlock() +
        getPointsFragmentShaderColoringBlock() +
        getFragmentShaderEndBlock( alphaSort );
}

} // namespace MR
