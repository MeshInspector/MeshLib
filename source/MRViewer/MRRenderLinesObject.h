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

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;
    virtual size_t heapBytes() const override;

    // requested line width clamped to the range of hardware supported values
    float actualLineWidth() const;

private:
    const ObjectLinesHolder* objLines_ = nullptr;

    /// ...
    mutable RenderObjectBuffer bufferObj_;
    mutable int vertPosSize_{ 0 };
    mutable int vertNormalsSize_{ 0 };
    mutable int vertColorsSize_{ 0 };
    mutable int vertUVSize_{ 0 };
    mutable int lineIndicesSize_{ 0 };

    RenderBufferRef<Vector3f> loadVertPosBuffer_() const;
    RenderBufferRef<Vector3f> loadVertNormalsBuffer_() const;
    RenderBufferRef<Color> loadVertColorsBuffer_() const;
    RenderBufferRef<UVCoord> loadVertUVBuffer_() const;
    RenderBufferRef<Vector2i> loadLineIndicesBuffer_() const;

    typedef unsigned int GLuint;

    GLuint linesArrayObjId_{ 0 };
    GLuint linesPickerArrayObjId_{ 0 };

    mutable GlBuffer vertPosBuffer_;
    mutable GlBuffer vertUVBuffer_;
    mutable GlBuffer vertNormalsBuffer_;
    mutable GlBuffer vertColorsBuffer_;

    mutable GlBuffer lineIndicesBuffer_;
    GLuint texture_{ 0 };

    GLuint pointsSelectionTex_{ 0 };
    GLuint lineColorsTex_{ 0 };

    void bindLines_() const;
    void bindLinesPicker_() const;

    void drawPoints_( const RenderParams& params ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
};

// Returns the range of line widths that are allowed by current renderer
MRVIEWER_API const Vector2f& GetAvailableLineWidthRange();

}