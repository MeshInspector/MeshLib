#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRRenderGLHelpers.h"

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

    std::vector<VertId> validIndicesBufferObj_;
    std::vector<unsigned> vertSelectionTexture_;

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