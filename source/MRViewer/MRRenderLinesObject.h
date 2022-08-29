#pragma once
#include "exports.h"
#include <MRMesh/MRIRenderObject.h>
#include <MRMesh/MRMeshTexture.h>
#include <MRMesh/MRId.h>
#include "MRRenderGLHelpers.h"

namespace MR
{
class RenderLinesObject : public IRenderObject
{
public:
    RenderLinesObject( const VisualObject& visObj );
    ~RenderLinesObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;

    // requested line width clamped to the range of hardware supported values
    float actualLineWidth() const;

private:
    const ObjectLinesHolder* objLines_ = nullptr;

    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    std::vector<Vector3f> vertPosBufferObj_;
    std::vector<Vector3f> vertNormalsBufferObj_;
    std::vector<Color> vertColorsBufferObj_;
    std::vector<UVCoord> vertUVBufferObj_;
    std::vector<Vector2i> linesIndicesBufferObj_;

    typedef unsigned int GLuint;

    GLuint linesArrayObjId_{ 0 };
    GLuint linesPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertUVBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;

    GlBuffer lineIndicesBuffer_;
    GLuint texture_{ 0 };

    GLuint pointsSelectionTex_{ 0 };
    GLuint lineColorsTex_{ 0 };

    void bindLines_();
    void bindLinesPicker_();

    void drawPoints_( const RenderParams& params );

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_;
};

// Returns the range of line widths that are allowed by current renderer
MRVIEWER_API const Vector2f& GetAvailableLineWidthRange();

}