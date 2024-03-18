#pragma once
#include "exports.h"
#include <MRMesh/MRIRenderObject.h>
#include <MRMesh/MRMeshTexture.h>
#include <MRMesh/MRId.h>
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderLinesObject : public virtual IRenderObject
{
public:
    RenderLinesObject( const VisualObject& visObj );
    ~RenderLinesObject();

    virtual bool render( const ModelRenderParams& params ) override;
    virtual void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;
    virtual size_t glBytes() const override;
    virtual void forceBindAll() override;

private:
    const ObjectLinesHolder* objLines_ = nullptr;
    int lineIndicesSize_{ 0 };

    typedef unsigned int GLuint;

    GLuint linesArrayObjId_{ 0 };
    GLuint linesPickerArrayObjId_{ 0 };

    GlTexture2 positionsTex_;
    GlTexture2 vertColorsTex_;
    GlTexture2 lineColorsTex_;

    void render_( const ModelRenderParams& params, bool points );
    void renderPicker_( const ModelBaseRenderParams& params, unsigned geomId, bool points );

    void bindPositions_( GLuint shaderId );

    void bindLines_( GLStaticHolder::ShaderType shaderType );
    void bindLinesPicker_( GLStaticHolder::ShaderType shaderType );

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
