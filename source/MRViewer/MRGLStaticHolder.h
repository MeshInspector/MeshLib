#pragma once
#include "MRRenderHelpers.h"
#include "MRMesh/MRLog.h"
#include <array>

namespace MR
{
// This class holds all shaders that are used in the program
// creates shader on access if it is needed
// holds shared memory buffer for loading to GPU
class GLStaticHolder
{
public:
    typedef unsigned int GLuint;
    enum ShaderType
    {
        DrawMesh,
        Picker,
        TransparentMesh,
        MeshBorder,
        TransparentMeshBorder,

        DrawPoints,
        DrawLines,

        Labels,

        AdditionalLines,
        AdditionalPoints,
        AdditionalPointsNoOffset, // special shader for old intel gpu (Intel HD 4000)
        AdditionalQuad,
        ViewportBorder,
        TransparencyOverlayQuad,
        ShadowOverlayQuad,
        SimpleOverlayQuad,
        Count
    };

    // Creates shader if it is not and return valid id
    static GLuint getShaderId( ShaderType type );
    // Free shader from GL
    static void freeShader( ShaderType type );
    // Free all shaders from GL
    static void freeAllShaders();
    // Memory buffer for objects that about to be loaded to GPU, shared among different data types
    static RenderObjectBuffer& getStaticGLBuffer();
private:
    GLStaticHolder();
    ~GLStaticHolder();

    static GLStaticHolder& instance_();

    void createShader_( ShaderType type );

    std::array<GLuint, size_t( Count )> shadersIds_;

    // it is stored here to prolong its life till this destructor
    std::shared_ptr<spdlog::logger> logger_;

    RenderObjectBuffer glBuffer_;
};
}