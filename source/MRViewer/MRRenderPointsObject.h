#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderPointsObject : public IRenderObject
{
public:
    RenderPointsObject( const VisualObject& visObj );
    ~RenderPointsObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;
    virtual size_t glBytes() const override;
    virtual void forceBindAll() override;
private:
    const ObjectPointsHolder* objPoints_;

    int validIndicesSize_{ 0 };
    Vector2i vertSelectionTextureSize_;

    RenderBufferRef<VertId> loadValidIndicesBuffer_();
    RenderBufferRef<unsigned> loadVertSelectionTextureBuffer_();

    typedef unsigned int GLuint;
    GLuint pointsArrayObjId_{ 0 };
    GLuint pointsPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;
    GlBuffer validIndicesBuffer_;

    GlTexture2 vertSelectionTex_;

    int maxTexSize_{ 0 };

    void bindPoints_();
    void bindPointsPicker_();

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    bool hasNormalsBackup_{ false };

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_;
};

}
