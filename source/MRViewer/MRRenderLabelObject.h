#pragma once
#include "MRMesh/MRIRenderObject.h"

namespace MR
{
class RenderLabelObject : public IRenderObject
{
public:
    RenderLabelObject( const VisualObject& visObj );
    ~RenderLabelObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;

private:
    const ObjectLabel* objLabel_;

    mutable std::vector<Vector3i> facesIndicesBufferObj_;
    mutable std::vector<Vector3i> bgFacesIndicesBufferObj_;
    typedef unsigned int GLuint;

    GLuint labelArrayObjId_{ 0 };
    GLuint vertPosBufferObjId_{ 0 };
    GLuint facesIndicesBufferObjId_{ 0 };

    GLuint pointArrayObjId_{ 0 };
    GLuint pointBufferObjId_{ 0 };
    GLuint validIndicesBufferObjId_{ 0 };

    GLuint bgArrayObjId_{ 0 };
    GLuint bgVertPosBufferObjId_{ 0 };
    GLuint bgFacesIndicesBufferObjId_{ 0 };

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
};

}