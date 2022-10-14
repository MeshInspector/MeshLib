#include "MRGLStaticHolder.h"
#include "MRCreateShader.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"

#ifndef __EMSCRIPTEN__
#define MR_GLSL_VERSION_LINE R"(#version 150)"
#else
#define MR_GLSL_VERSION_LINE R"(#version 300 es)"
#endif

namespace
{
std::string getShaderName( MR::GLStaticHolder::ShaderType type )
{
    const std::array<std::string, size_t( MR::GLStaticHolder::Count )> names =
    {
        "Mesh shader",
        "Picker shader",
        "Alpha-sort mesh shader",
        "Mesh border shader",
        "Alpha-sort mesh border shader",
        "Points shader",
        "Lines shader",
        "Labels shader",
        "Viewport lines shader",
        "Viewport points shader",
        "Viewport points shader (no offset)",
        "Immediate tri shader",
        "Viewport border shader",
        "Alpha-sort overlay shader",
        "Shadow overlay shader",
        "Simple overlay shader"
    };
    return names[type];
}
}

namespace MR
{

unsigned GLStaticHolder::getShaderId( ShaderType type )
{
    auto& instance = GLStaticHolder::instance_();
    auto& id = instance.shadersIds_[type];
    if ( id == 0 )
    {
        instance.createShader_( type );
    }

    return id;
}

void GLStaticHolder::freeShader( ShaderType type )
{
    auto& instance = GLStaticHolder::instance_();
    if ( instance.shadersIds_[type] == 0 )
        return;

    destroyShader( instance.shadersIds_[type] );
    instance.shadersIds_[type] = 0;
}

void GLStaticHolder::freeAllShaders()
{
    for ( int i = 0; i < ShaderType::Count; ++i )
        freeShader( ShaderType( i ) );
}

GLStaticHolder::GLStaticHolder()
{
    logger_ = Logger::instance().getSpdLogger();
    for ( int i = 0; i < ShaderType::Count; ++i )
        shadersIds_[i] = 0;
}

GLStaticHolder::~GLStaticHolder()
{
    for ( int i = 0; i < ShaderType::Count; ++i )
        if ( shadersIds_[i] != 0 )
            logger_->warn( "{} is not freed", getShaderName( ShaderType( i ) ) );
}

GLStaticHolder& GLStaticHolder::instance_()
{
    static GLStaticHolder instance;
    return instance;
}

void GLStaticHolder::createShader_( ShaderType type )
{
    std::string vertexShader;
    std::string fragmentShader;
    if ( type == DrawMesh || type == TransparentMesh )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  in vec3 position;       // (in from base) vert position
  in vec3 normal;         // (in from base) vert normal
  in vec4 K;              // (in from base) vert color
  in vec2 texcoord;       // (in from base) vert uv coordinate

  out vec2 texcoordi;    // (out to fragment shader) vert uv coordinate
  out vec3 world_pos;    // (out to fragment shader) vert transformed position
  out vec4 Ki;           // (out to fragment shader) vert color 
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)
  flat out highp uint primitiveId;

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    position_eye = vec3 (view * vec4 (world_pos, 1.0));
    normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
    normal_eye = normalize(normal_eye);
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    Ki = K;
    texcoordi = texcoord;
    primitiveId = uint(gl_VertexID) / 3u;
  }
)";
        if ( type == DrawMesh )
        {
            int major, minor;
            auto window = glfwGetCurrentContext();
            major = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MAJOR );
            minor = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MINOR );

            if ( major < 4 || ( major == 4 && minor < 3 ) )
            {
                fragmentShader =
                    MR_GLSL_VERSION_LINE R"(
                    precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform sampler2D faceNormals;     // (in from base) normals per face
  uniform sampler2D faceColors;      // (in from base) face color
  uniform bool perFaceColoring;      // (in from base) use faces colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool flatShading;          // (in from base) linear interpolate normals if false
  uniform bool onlyOddFragments;     // (in from base) discard every second fragment
  uniform bool showSelectedFaces;    // (in from base) use selection or not
 
  uniform vec4 mainColor;            // (in from base) main color
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 backColor;            // (in from base) back face color
  uniform vec4 selectionBackColor;   // (in from base) selection back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true
  uniform sampler2D tex;             // (in from base) texture
  uniform float specular_exponent;   // (in from base) lighting parameter 
  uniform bool useTexture;           // (in from base) enable texture
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
                                     
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec2 texcoordi;                 // (in from vertex shader) vert uv coordinate
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  flat in highp uint primitiveId;
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    if (onlyOddFragments && ((int(gl_FragCoord.x) + int(gl_FragCoord.y)) % 2) == 1)
      discard;

    vec3 normEyeCpy = normal_eye;
    if ( flatShading )
    {
      ivec2 texSize = textureSize( faceNormals, 0 );
      vec3 norm = vec3( texelFetch( faceNormals, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 ) );
      normEyeCpy = normalize(vec3 (normal_matrix * vec4 (norm, 0.0)));
    }
    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));

    vec4 colorCpy;
    bool selected = false;
    if ( showSelectedFaces )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }

    if ( gl_FrontFacing == invertNormals )
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selectionBackColor;
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perFaceColoring )
        {
            ivec2 texSize = textureSize( faceColors, 0 );
            colorCpy = texelFetch(faceColors, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 );
        }
        else
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if ( useTexture )
    {
      vec4 textColor = texture(tex, texcoordi);
      float destA = colorCpy.a;
      colorCpy.a = textColor.a + destA * ( 1.0 - textColor.a );
      if ( colorCpy.a == 0.0 )
        colorCpy.rgb = vec3(0);
      else
        colorCpy.rgb = mix(colorCpy.rgb*destA,textColor.rgb,textColor.a)/colorCpy.a;
    }  

    if (gl_FrontFacing == false) // don't use !gl_FrontFacing for some rare mac issue
      dot_prod = -dot_prod;

    if (dot_prod < 0.0)
      dot_prod = 0.0;

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if (dot_prod_specular < 0.0)
      dot_prod_specular = 0.0;

    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a);

    if (outColor.a == 0.0)
      discard;
  }
)";
            }
            else
            {
                fragmentShader =
                    R"(#version 430 core
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform sampler2D faceNormals;     // (in from base) normals per face
  uniform sampler2D faceColors;      // (in from base) face color
  uniform bool perFaceColoring;      // (in from base) use faces colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool flatShading;          // (in from base) linear interpolate normals if false
  uniform bool onlyOddFragments;     // (in from base) discard every second fragment
  uniform bool showSelectedFaces;    // (in from base) use selection or not

  uniform vec4 mainColor;            // (in from base) main color
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 backColor;            // (in from base) back face color
  uniform vec4 selectionBackColor;   // (in from base) selection back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true
  uniform sampler2D tex;             // (in from base) texture
  uniform float specular_exponent;   // (in from base) lighting parameter 
  uniform bool useTexture;           // (in from base) enable texture
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
                                     
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec2 texcoordi;                 // (in from vertex shader) vert uv coordinate
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  flat in highp uint primitiveId;
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    gl_SampleMask[0] = gl_SampleMaskIn[0];
    if (onlyOddFragments )
    {
      if (  mod( gl_FragCoord.x + gl_FragCoord.y, 2 ) < 1 )
        gl_SampleMask[0] = gl_SampleMaskIn[0] & 0xaaaaaaaa;
      else
        gl_SampleMask[0] = gl_SampleMaskIn[0] & 0x55555555;
    }

    vec3 normEyeCpy = normal_eye;
    if ( flatShading )
    {
      ivec2 texSize = textureSize( faceNormals, 0 );
      vec3 norm = vec3( texelFetch( faceNormals, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 ) );
      normEyeCpy = normalize(vec3 (normal_matrix * vec4 (norm, 0.0)));
    }
    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
      
    vec4 colorCpy;
    bool selected = false;
    if ( showSelectedFaces )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }
    if ( gl_FrontFacing == invertNormals )
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selectionBackColor;
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perFaceColoring )
        {
            ivec2 texSize = textureSize( faceColors, 0 );
            colorCpy = texelFetch(faceColors, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 );
        }
        else 
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if ( useTexture )
    {
      vec4 textColor = texture(tex, texcoordi);
      float destA = colorCpy.a;
      colorCpy.a = textColor.a + destA * ( 1.0 - textColor.a );
      if ( colorCpy.a == 0.0 )
        colorCpy.rgb = vec3(0);
      else
        colorCpy.rgb = mix(colorCpy.rgb*destA,textColor.rgb,textColor.a)/colorCpy.a;
    }  

    if (gl_FrontFacing == false) // don't use !gl_FrontFacing for some rare mac issue
      dot_prod = -dot_prod;

    dot_prod = max(dot_prod,0.0);

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    dot_prod_specular = max(dot_prod_specular,0);
    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a);

    if (outColor.a == 0.0)
      discard;
  }
)";
            }
        }
        else
        {
            fragmentShader =
                R"(#version 430 core

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

  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform sampler2D faceNormals;     // (in from base) normals per face
  uniform sampler2D faceColors;      // (in from base) face color
  uniform bool perFaceColoring;      // (in from base) use faces colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool flatShading;          // (in from base) linear interpolate normals if false
  uniform bool onlyOddFragments;     // (in from base) discard every second fragment
  uniform bool showSelectedFaces;    // (in from base) use selection or not

  uniform vec4 mainColor;            // (in from base) main color
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 backColor;            // (in from base) back face color
  uniform vec4 selectionBackColor;   // (in from base) selection back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true
  uniform sampler2D tex;             // (in from base) texture
  uniform float specular_exponent;   // (in from base) lighting parameter 
  uniform bool useTexture;           // (in from base) enable texture
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
                                     
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec2 texcoordi;                 // (in from vertex shader) vert uv coordinate
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  flat in highp uint primitiveId;
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    if (onlyOddFragments && mod(gl_FragCoord.x + gl_FragCoord.y, 2) < 1)
      discard;

    vec3 normEyeCpy = normal_eye;
    if ( flatShading )
    {
      ivec2 texSize = textureSize( faceNormals, 0 );
      vec3 norm = vec3( texelFetch( faceNormals, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 ) );
      normEyeCpy = normalize(vec3 (normal_matrix * vec4 (norm, 0.0)));
    }
    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
      
    vec4 colorCpy;
    bool selected = false;
    if ( showSelectedFaces )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }
    if ( gl_FrontFacing == invertNormals )
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selectionBackColor;
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perFaceColoring )
        {
            ivec2 texSize = textureSize( faceColors, 0 );
            colorCpy = texelFetch(faceColors, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 );
        }
        else 
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if ( useTexture )
    {
      vec4 textColor = texture(tex, texcoordi);
      float destA = colorCpy.a;
      colorCpy.a = textColor.a + destA * ( 1.0 - textColor.a );
      if ( colorCpy.a == 0.0 )
        colorCpy.rgb = vec3(0);
      else
        colorCpy.rgb = mix(colorCpy.rgb*destA,textColor.rgb,textColor.a)/colorCpy.a;
    }  

    if (gl_FrontFacing == false) // don't use !gl_FrontFacing for some rare mac issue
      dot_prod = -dot_prod;

    dot_prod = max(dot_prod,0.0);

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    dot_prod_specular = max(dot_prod_specular,0);
    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a);

    if (outColor.a == 0.0)
      discard;

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
        }
    }
    else if ( type == Picker )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform uint primBucketSize;
  uniform float pointSize;

  in vec3 position;
  
  out vec3 world_pos;
  flat out highp uint primitiveId;

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    gl_Position = proj * view * vec4 (world_pos, 1.0); //proj * view * vec4(position, 1.0);"
    primitiveId = uint(gl_VertexID) / primBucketSize;
    gl_PointSize = pointSize;
  }
)";

        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform bool useClippingPlane;
  uniform vec4 clippingPlane;
  uniform uint uniGeomId;

  in vec3 world_pos;
  flat in highp uint primitiveId;

  out highp uvec4 color;

  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    color.r = primitiveId;

    color.g = uniGeomId;

    color.a = uint(gl_FragCoord.z * 4294967295.0);
  }
)";
    }
    else if ( type == DrawPoints || type == DrawLines )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;
  uniform float pointSize;
  uniform uint primBucketSize;

  in vec3 position;       // (in from base) vert position
  in vec3 normal;         // (in from base) vert normal
  in vec4 K;              // (in from base) vert color

  out vec3 world_pos;    // (out to fragment shader) vert transformed position
  out vec4 Ki;           // (out to fragment shader) vert color 
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)
  flat out highp uint primitiveId;

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    position_eye = vec3 (view * vec4 (world_pos, 1.0));
    normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
    normal_eye = normalize(normal_eye);
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    Ki = K;
    gl_PointSize = pointSize;
    primitiveId = uint(gl_VertexID) / primBucketSize;
  }
)";
        if ( type == DrawPoints )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform bool showSelectedVertices;    // (in from base) use selection or not
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 selectionBackColor;   // (in from base) selection back face color

  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool hasNormals;           // (in from base) dont use normals if they are not
 
  uniform vec4 mainColor;            // (in from base) color if colormap is off
  uniform vec4 backColor;            // (in from base) back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true

  uniform float specular_exponent;   // (in from base) lighting parameter
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
                                     
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position

  flat in highp uint primitiveId;
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    vec3 normEyeCpy = normal_eye;
    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    if (!hasNormals)
      normEyeCpy = direction_to_light_eye;

    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
    
    vec4 colorCpy;
    bool selected = false;
    if ( showSelectedVertices )
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
            colorCpy = selectionBackColor;
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
    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a);
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else // DrawLines
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform sampler2D lineColors;  // (in from base) line color
  uniform bool perLineColoring;      // (in from base) use lines colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool hasNormals;           // (in from base) dont use normals if they are not
 
  uniform vec4 mainColor;            // (in from base) color if colormap is off
  uniform vec4 backColor;            // (in from base) back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true

  uniform float specular_exponent;   // (in from base) lighting parameter 
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
  flat in highp uint primitiveId;
                                     
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    vec3 normEyeCpy = normal_eye;
    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    if (!hasNormals)
      normEyeCpy = direction_to_light_eye;

    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
      
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
    bool frontFacing = dot_prod >= 0.0;
    if ( frontFacing == invertNormals )
    {
       colorCpy = backColor;
    }
    if (!frontFacing)
      dot_prod = -dot_prod;

    if (dot_prod < 0.0)
      dot_prod = 0.0;

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if (dot_prod_specular < 0.0)
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a);
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
    }
    else if ( type == MeshBorder || type == TransparentMeshBorder )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 model;
  uniform vec4 uniformColor;

  in vec3 position;
  out vec4 color_frag;
  out vec3 world_pos;    // (out to fragment shader) vert transformed position

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    gl_Position = proj * view * vec4(world_pos, 1.0);
    color_frag = uniformColor;
  }
)";
        if ( type == MeshBorder )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane

  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else
        {
            fragmentShader =
                R"(#version 430 core
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

  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane

  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;
    outColor = color_frag;

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
        }
    }
    else if ( type == Labels )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec3 basePos;
  uniform vec2 modifier;
  uniform vec2 shift;

  in vec3 position;       // (in from base) vert position

  void main()
  {
    vec4 projBasePos = proj * ( view * model * vec4( basePos, 1.0 ) );
    vec4 coord = projBasePos + projBasePos.w * vec4( modifier.x * (position.x - shift.x), modifier.y * (position.y - shift.y), 0.0, 0.0);
    gl_Position = coord / coord.w;
  }
)";
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;

  uniform vec4 mainColor;            // (in from base) main color
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    outColor = mainColor;

    if (outColor.a == 0.0)
      discard;
  }
)";
    }
    else
    {
        if ( type == AdditionalLines || type == AdditionalPoints || type == AdditionalPointsNoOffset )
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform mat4 view;
  uniform mat4 proj;
  uniform float pointSize;

  in vec3 position;
  in vec4 color;
  out vec4 color_frag;

  void main()
  {
    gl_Position = proj * view * vec4 (position, 1.0);
    color_frag = color;
    gl_PointSize = pointSize;
  }
)";
        }
        else if ( type == AdditionalQuad )
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform mat4 view;
  uniform mat4 proj;

  in vec3 position;
  in vec3 normal;         // (in from base) vert normal
  in vec4 color;

  out vec4 color_frag;
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)

  void main()
  {
    position_eye = vec3 (view * vec4 (position, 1.0));
    normal_eye = normalize(vec3 (view * vec4 (normal, 0.0)));
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    color_frag = color;
  }
)";
        }
        else
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform vec4 user_color;
  in vec3 position;
  out vec4 color_frag;

  void main()
  {
    gl_Position = vec4 (position, 1.0);
    color_frag = user_color;
  }
)";
        }

        if ( type == AdditionalLines || type == ViewportBorder )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform float offset;
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
    gl_FragDepth = gl_FragCoord.z + offset;
  }
)";
        }
        else if ( type == AdditionalPoints )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform float offset;

  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
    gl_FragDepth = gl_FragCoord.z + offset;
  }
)";
        }
        else if ( type == AdditionalPointsNoOffset )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == AdditionalQuad )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec3 light_position_eye;   // (in from base) light position transformed by view only (not proj)
                                     
  float specular_exponent = 35.0f;
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 color_frag;                        // (in from vertex shader) vert color
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {    
    vec3 vector_to_light_eye = light_position_eye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = abs(dot (direction_to_light_eye, normalize(normal_eye)));

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if ( dot_prod_specular < 0.0 )
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specular_exponent);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(color_frag);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,color_frag.a);

    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == TransparencyOverlayQuad )
        {
            fragmentShader =
                R"(
#version 430 core

#define MAX_FRAGMENTS 18

// made as macro to have it inline
#define swapNodes(a, b) \
{ \
	Node tmp = a; \
	a = b; \
	b = tmp; \
}

layout (early_fragment_tests) in;

out vec4 color;

struct Node 
{
    vec4    color;
    float   depth;
    uint    next;
};

layout (binding = 0, r32ui)      uniform uimage2D    heads;
layout (binding = 0, offset = 0) uniform atomic_uint numNodes;

layout (binding = 0, std430 ) buffer Lists
{
    Node nodes [];
};

void sortedInsert(inout Node frags[MAX_FRAGMENTS], Node node, inout int count) 
{
	if (count == MAX_FRAGMENTS) 
	{
		if (node.depth < frags[count - 1].depth) 
		{
			swapNodes(node, frags[count - 1]);
		}
	}
	else 
	{
		frags[count] = node;
		count++;
	}

	uint i = count - 1;
	while (i > 0 && frags[i].depth < frags[i - 1].depth)
	{
		swapNodes(frags[i], frags[i - 1]);
		i--;
	}
}

void main(void)
{
    uint nodesCounter = atomicCounter( numNodes );
    if (nodesCounter == 0)
      discard;

    Node frags [MAX_FRAGMENTS];
    int count = 0;
    // get the index of the head of the list
    uint n = imageLoad ( heads, ivec2 ( gl_FragCoord.xy ) ).r;

    // sort linked list to array
    while ( n != 0xFFFFFFFF )
    {
        Node node = nodes[n];
        sortedInsert(frags, node, count);
        n = node.next;
    }
    
    if (count == 0)
      discard;
    
    // traverse the array, and combine the colors using the alpha channel    
    color = vec4(0, 0, 0, 0);
    
    for ( int i = count-1; i >= 0; i-- )
    {
        float destA = color.a;
        color.a = frags[i].color.a + destA * ( 1.0 - frags[i].color.a );
        if ( color.a == 0.0 )
          color.rgb = vec3(0);
        else
          color.rgb = mix(color.rgb*destA,frags[i].color.rgb,frags[i].color.a)/color.a;
    }
    
    gl_FragDepth = frags[count-1].depth;
}
)";
        }
        else if ( type == ShadowOverlayQuad )
        {
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform sampler2D pixels;
  uniform vec4 color;
  uniform vec2 shift;
  uniform float blurRadius;
  out vec4 outColor;                 // (out to render) fragment color

  #define NUM_DIRECTIONS 12
  #define QUALITY 3
  
  #define DIR_STEP 0.5235987756
  #define INV_QUALITY 0.33333334
  #define INV_NUM_SAMPLES 0.020833333

  void main()
  { 
    ivec2 texSize = textureSize( pixels, 0 );
    vec2 pos = gl_FragCoord.xy + shift;
    pos = vec2( pos.x/float(texSize.x),pos.y/float(texSize.y) );

    if ( texelFetch(pixels, ivec2( gl_FragCoord.xy ), 0 ).a == 1.0 )
      discard;

    if ( textureLod(pixels,pos,max(10.0,log2(blurRadius)+2.0)).a == 0.0 )
      discard;

    float avgValue = 0.0;
    float maxRadiusSq = blurRadius*blurRadius;

    avgValue = texture(pixels, pos).a;

    for ( int r = 1; r <= QUALITY; r = r + 1 )
    {
        float radius = float(r)*(blurRadius-0.5)*INV_QUALITY;
        vec2 normRad = vec2(radius/float(texSize.x),radius/float(texSize.y));
        for ( int ang = 0; ang < NUM_DIRECTIONS; ang = ang + 1 )
        {
            float realAng = float(ang)*DIR_STEP;
            vec2 addPos = vec2(cos(realAng)*normRad.x, sin(realAng)*normRad.y);
            avgValue = avgValue + texture(pixels, pos+addPos).a;
        }
    }
    avgValue = avgValue * INV_NUM_SAMPLES;
    outColor = vec4(color.rgb,avgValue*color.a);
    gl_FragDepth = 0.9999;
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == SimpleOverlayQuad )
        {
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
                precision highp float;
  uniform sampler2D pixels;
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  { 
    outColor = texelFetch(pixels, ivec2( gl_FragCoord.xy ), 0 );
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
    }

    DisabledWarnings warns = {};
    if ( type == TransparencyOverlayQuad )
        warns.push_back( 7050 );

    createShader( getShaderName( type ), vertexShader, fragmentShader, shadersIds_[type], warns );
}

RenderObjectBuffer &GLStaticHolder::getStaticGLBuffer()
{
    return instance_().glBuffer_;
}

}
