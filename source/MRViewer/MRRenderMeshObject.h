#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include <bitset>

namespace MR
{
class RenderMeshObject : public IRenderObject
{
public:
    RenderMeshObject( const VisualObject& visObj );
    ~RenderMeshObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;
    virtual size_t heapBytes() const override;

private:
    const ObjectMeshHolder* objMesh_;

    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    enum BufferType {
        VERTEX_POSITIONS,
        PICKER_VERTEX_POSITIONS,
        VERTEX_NORMALS,
        FACE_NORMALS,
        VERTEX_COLORMAPS,
        FACES,
        EDGES,
        VERTEX_UVS,
        FACE_SELECTION,
        BORDER_LINES,
        EDGE_SELECTION,

        BUFFER_COUNT,
    };
    mutable Buffer<std::byte> bufferObj_;
    mutable std::size_t elementSize_;
    mutable std::array<std::size_t, BUFFER_COUNT> elementCount_;
    mutable std::bitset<BUFFER_COUNT> elementDirty_;

    template <typename T>
    T* prepareBuffer_( BufferType type, std::size_t elementCount ) const
    {
        elementSize_ = sizeof(T);
        elementCount_[type] = elementCount;
        auto size = elementSize_ * elementCount_[type];
        if ( bufferObj_.size() < size )
            bufferObj_.resize( size );
        return reinterpret_cast<T*>( bufferObj_.data() );
    }

    void loadBuffer_( BufferType type ) const;

    typedef unsigned int GLuint;

    GLuint borderArrayObjId_{ 0 };
    GLuint borderBufferObjId_{ 0 };

    GLuint selectedEdgesArrayObjId_{ 0 };
    GLuint selectedEdgesBufferObjId_{ 0 };

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    mutable GlBuffer vertPosBuffer_;
    mutable GlBuffer vertUVBuffer_;
    mutable GlBuffer vertNormalsBuffer_;
    mutable GlBuffer vertColorsBuffer_;

    GLuint facesIndicesBufferObjId_{ 0 };
    GLuint edgesIndicesBufferObjId_{ 0 };
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    int maxTexSize_{ 0 };

    void renderEdges_( const RenderParams& parameters, GLuint vao, GLuint vbo, BufferType bufferType, const Color& color ) const;

    void renderMeshEdges_( const RenderParams& parameters ) const;

    void bindMesh_( bool alphaSort ) const;
    void bindMeshPicker_() const;

    void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_( ViewportId id ) const;

    void resetBuffers_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    mutable bool normalsBound_{ false };
    // ...
    mutable bool hasVertNormals_{ false };
};

}