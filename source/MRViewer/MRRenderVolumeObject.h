#pragma once
#include "MRViewerFwd.h"
#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRVoxelsFwd.h"

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class MRVIEWER_CLASS RenderVolumeObject : public virtual IRenderObject
{
public:
    RenderVolumeObject( const VisualObject& visObj );
    ~RenderVolumeObject();

    virtual bool render( const ModelRenderParams& params ) override;
    virtual void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;
    virtual size_t glBytes() const override;
    virtual void forceBindAll() override;

private:
    const ObjectVoxels* objVoxels_{ nullptr };

    typedef unsigned int GLuint;
    GLuint volumeArrayObjId_{ 0 };
    GlBuffer volumeVertsBuffer_;
    GlBuffer volumeIndicesBuffer_;

    GlTexture3 volume_;
    GlTexture2 denseMap_;

    Vector2i activeVoxelsTextureSize_;
    GlTexture2 activeVoxelsTex_;
    int maxTexSize_{ 0 };

    // When rendering for picker, `nonPickerParams` will be null and `geomId` will be zero.
    void render_( const ModelBaseRenderParams& params, const ModelRenderParams* nonPickerParams, unsigned geomId );
    void bindVolume_( bool picker );

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    RenderBufferRef<unsigned> loadActiveVoxelsTextureBuffer_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_{ 0 };
};

}
#endif
