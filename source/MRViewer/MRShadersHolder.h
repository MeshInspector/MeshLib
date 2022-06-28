#pragma once
#include "MRMesh/MRLog.h"
#include <array>

namespace MR
{
// This class holds all shaders that are used in the program
// creates shader on access if it is needed
class ShadersHolder
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
        Count
    };

    // Creates shader if it is not and return valid id
    static GLuint getShaderId( ShaderType type );
    // Free shader from GL
    static void freeShader( ShaderType type );
    // Free all shaders from GL
    static void freeAllShaders();
private:
    ShadersHolder();
    ~ShadersHolder();

    static ShadersHolder& instance_();
    
    void createShader_( ShaderType type );

    std::array<GLuint, size_t( Count )> shadersIds_;

    // it is stored here to prolong its life till this destructor
    std::shared_ptr<spdlog::logger> logger_;
};
}