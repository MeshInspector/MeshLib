#pragma once
#include "exports.h"
#include "MRViewportGL.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRIRenderObject.h"

namespace MR
{

namespace ImmediateGL
{

// Immediate render params for quads
struct TriRenderParams : BaseRenderParams
{
    bool depthTest{ true };
    Vector3f lightPos;
    Matrix4f modelMatrix; // transformation of each triangle in world space
};

struct TriCornerColors
{
    Vector4f a, b, c;
};

// Draw tris immediately (flat shaded)
MRVIEWER_API void drawTris( const std::vector<Triangle3f>& tris, const std::vector<TriCornerColors>& colors, const ImmediateGL::TriRenderParams& params );

}

}