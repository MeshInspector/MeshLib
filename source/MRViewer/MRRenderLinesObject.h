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

    // requested line width clamped to the range of hardware supported values
    float actualLineWidth() const;

private:
    const ObjectLinesHolder* objLines_ = nullptr;

    // memory buffer for objects that about to be loaded to GPU, shared among different data types
    RenderObjectBuffer bufferObj_;
    int vertPosSize_{ 0 };
    int vertNormalsSize_{ 0 };
    int vertColorsSize_{ 0 };
    int vertUVSize_{ 0 };
    int lineIndicesSize_{ 0 };

    RenderBufferRef<Vector3f> loadVertPosBuffer_() const;
    RenderBufferRef<Vector3f> loadVertNormalsBuffer_() const;
    RenderBufferRef<Color> loadVertColorsBuffer_() const;
    RenderBufferRef<UVCoord> loadVertUVBuffer_() const;
    RenderBufferRef<Vector2i> loadLineIndicesBuffer_() const;

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
