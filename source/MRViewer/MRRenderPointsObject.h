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

private:
    const ObjectPointsHolder* objPoints_;

    // memory buffer for objects that about to be loaded to GPU, shared among different data types
    RenderObjectBuffer bufferObj_;
    int validIndicesSize_{ 0 };
    int vertSelectionTextureSize_{ 0 };

    RenderBufferRef<VertId> loadValidIndicesBuffer_() const;
    RenderBufferRef<unsigned> loadVertSelectionTextureBuffer_() const;

    typedef unsigned int GLuint;
    GLuint pointsArrayObjId_{ 0 };
    GLuint pointsPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;

    GlBuffer validIndicesBuffer_;

    GLuint vertSelectionTex_{ 0 };

    void bindPoints_();
    void bindPointsPicker_();

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_;
};

}
