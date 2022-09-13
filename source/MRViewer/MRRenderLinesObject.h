#pragma once
#include "exports.h"
#include <MRMesh/MRIRenderObject.h>
#include <MRMesh/MRMeshTexture.h>
#include <MRMesh/MRId.h>
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

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
    virtual size_t glBytes() const override;
    virtual void forceBindAll() override;
    // requested line width clamped to the range of hardware supported values
    float actualLineWidth() const;

private:
    const ObjectLinesHolder* objLines_ = nullptr;

    int vertPosSize_{ 0 };
    int vertNormalsSize_{ 0 };
    int vertColorsSize_{ 0 };
    int vertUVSize_{ 0 };
    int lineIndicesSize_{ 0 };

    RenderBufferRef<Vector3f> loadVertPosBuffer_();
    RenderBufferRef<Vector3f> loadVertNormalsBuffer_();
    RenderBufferRef<Color> loadVertColorsBuffer_();
    RenderBufferRef<UVCoord> loadVertUVBuffer_();
    RenderBufferRef<Vector2i> loadLineIndicesBuffer_();

    typedef unsigned int GLuint;

    GLuint linesArrayObjId_{ 0 };
    GLuint linesPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertUVBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;
    GlBuffer lineIndicesBuffer_;

    GlTexture2 texture_;
    GlTexture2 pointsSelectionTex_;
    GlTexture2 lineColorsTex_;

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
