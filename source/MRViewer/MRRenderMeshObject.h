#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"

namespace MR
{
class RenderMeshObject : public IRenderObject
{
public:
    RenderMeshObject( const VisualObject& visObj );
    ~RenderMeshObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;

private:
    const ObjectMeshHolder* objMesh_;

    // memory buffer for objects that about to be loaded to GPU
    Buffer<std::byte> bufferObj_;

    using DirtyFlag = uint32_t;

    template <DirtyFlag>
    struct BufferTypeHelper;
    template <DirtyFlag dirtyFlag>
    using BufferType = typename BufferTypeHelper<dirtyFlag>::type;

    std::array<std::size_t, 8 * sizeof( DirtyFlag )> bufferGLSize_; // in bits
    template <DirtyFlag>
    std::size_t& getGLSize_();
    template <DirtyFlag>
    std::size_t getGLSize_() const;

    template <typename T>
    class BufferRef;
    template <DirtyFlag dirtyFlag>
    BufferRef<BufferType<dirtyFlag>> prepareBuffer_( std::size_t glSize, DirtyFlag flagToReset = dirtyFlag );
    template <DirtyFlag dirtyFlag>
    BufferRef<BufferType<dirtyFlag>> loadBuffer_();

    typedef unsigned int GLuint;

    GLuint borderArrayObjId_{ 0 };
    GLuint borderBufferObjId_{ 0 };

    GLuint selectedEdgesArrayObjId_{ 0 };
    GLuint selectedEdgesBufferObjId_{ 0 };

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertUVBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;

    GlBuffer facesIndicesBuffer_;
    GlBuffer edgesIndicesBuffer_;
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    int maxTexSize_{ 0 };

    template <DirtyFlag>
    void renderEdges_( const RenderParams& parameters, GLuint vao, GLuint vbo, const Color& color );

    void renderMeshEdges_( const RenderParams& parameters );

    void bindMesh_( bool alphaSort );
    void bindMeshPicker_();

    void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_( ViewportId id );

    void resetBuffers_();

    // Marks dirty buffers that need to be uploaded to OpenGL
    DirtyFlag dirty_;
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    bool normalsBound_{ false };
};

}