#pragma once
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRVector2.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderLabelObject : public IRenderObject
{
public:
    RenderLabelObject( const VisualObject& visObj );
    ~RenderLabelObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;

    virtual size_t heapBytes() const override;

private:
    const ObjectLabel* objLabel_;

    int faceIndicesSize_{ 0 };

    RenderBufferRef<Vector3i> loadFaceIndicesBuffer_();

    typedef unsigned int GLuint;

    GLuint labelArrayObjId_{ 0 };
    GlBuffer vertPosBuffer_;
    GlBuffer facesIndicesBuffer_;

    GLuint srcArrayObjId_{ 0 };
    GlBuffer srcVertPosBuffer_;
    GlBuffer srcIndicesBuffer_;
    GLuint srcIndicesSelectionTexId_{ 0 };

    GLuint bgArrayObjId_{ 0 };
    GlBuffer bgVertPosBuffer_;
    GlBuffer bgFacesIndicesBuffer_;

    GLuint llineArrayObjId_{ 0 };
    GlBuffer llineVertPosBuffer_;
    GlBuffer llineEdgesIndicesBuffer_;

    void renderSourcePoint_( const RenderParams& renderParams );
    void renderBackground_( const RenderParams& renderParams );
    void renderLeaderLine_( const RenderParams& renderParams );

    void bindLabel_();

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_;
    bool dirtySrc_;
    bool dirtyBg_;
    bool dirtyLLine_;
    Vector3f positionState_;
    Vector2f pivotPointState_;
    float backgroundPaddingState_;
};

}
