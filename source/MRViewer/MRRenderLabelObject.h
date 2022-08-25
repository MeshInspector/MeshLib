#pragma once
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRVector2.h"
#include "MRRenderGLHelpers.h"

namespace MR
{
class RenderLabelObject : public IRenderObject
{
public:
    RenderLabelObject( const VisualObject& visObj );
    ~RenderLabelObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;

    virtual size_t heapBytes() const override;

private:
    const ObjectLabel* objLabel_;

    mutable std::vector<Vector3i> facesIndicesBufferObj_;
    typedef unsigned int GLuint;

    GLuint labelArrayObjId_{ 0 };
    mutable GlBuffer vertPosBuffer_;
    mutable GlBuffer facesIndicesBuffer_;

    GLuint srcArrayObjId_{ 0 };
    mutable GlBuffer srcVertPosBuffer_;
    mutable GlBuffer srcIndicesBuffer_;
    GLuint srcIndicesSelectionTexId_{ 0 };

    GLuint bgArrayObjId_{ 0 };
    mutable GlBuffer bgVertPosBuffer_;
    mutable GlBuffer bgFacesIndicesBuffer_;

    GLuint llineArrayObjId_{ 0 };
    mutable GlBuffer llineVertPosBuffer_;
    mutable GlBuffer llineEdgesIndicesBuffer_;

    void renderSourcePoint_( const RenderParams& renderParams ) const;
    void renderBackground_( const RenderParams& renderParams ) const;
    void renderLeaderLine_( const RenderParams& renderParams ) const;

    void bindLabel_() const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
    mutable bool dirtySrc_;
    mutable bool dirtyBg_;
    mutable bool dirtyLLine_;
    mutable Vector3f positionState_;
    mutable Vector2f pivotPointState_;
    mutable float backgroundPaddingState_;
};

}