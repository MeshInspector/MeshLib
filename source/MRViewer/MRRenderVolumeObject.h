#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderVolumeObject : public IRenderObject
{
public:
    RenderVolumeObject( const VisualObject& visObj );
    ~RenderVolumeObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;
    virtual size_t glBytes() const override;
    virtual void forceBindAll() override;
private:
    const ObjectVoxels* objVoxels_;

    typedef unsigned int GLuint;
    GLuint volumeArrayObjId_{ 0 };
    GLuint volumeBufferObjId_{ 0 };

    GlTexture3 volume_;
    GlTexture2 denseMap_;

    void bindVolume_();

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_{ 0 };
};

}
