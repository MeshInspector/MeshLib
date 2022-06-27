#pragma once
#include "exports.h"
#include "MRViewportGL.h"
#include "MRMesh/MRVector3.h"

namespace MR
{

namespace ImmediateGL
{

// Immediate render params for lines and points
struct RenderParams : ViewportGL::BaseRenderParams
{
    float width{1.0f};
    bool depthTest{ true };
};

// Immediate render params for quads
struct TriRenderParams : ViewportGL::BaseRenderParams
{
    bool depthTest{ true };
    Vector3f lightPos;
};

// Draw lines immediately
MRVIEWER_API void drawLines( const std::vector<LineSegm3f>& lines, const std::vector<SegmEndColors>& colors, const ImmediateGL::RenderParams& params );

// Draw points immediately
MRVIEWER_API void drawPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors, const ImmediateGL::RenderParams& params );

struct Tri
{
    Vector3f a, b, c;
};

struct TriCornerColors
{
    Vector4f a, b, c;
};

// Draw tris immediately (flat shaded)
MRVIEWER_API void drawTris( const std::vector<Tri>& tris, const std::vector<TriCornerColors>& colors, const ImmediateGL::TriRenderParams& params );

}

}