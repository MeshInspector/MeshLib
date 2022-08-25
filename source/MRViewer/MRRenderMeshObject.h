#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include <bitset>
#include <optional>

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

    template <BufferType>
    struct ElementType;

    mutable std::array<std::size_t, BUFFER_COUNT> elementCount_;
    mutable std::bitset<BUFFER_COUNT> elementDirty_;

    template <typename T>
    class BufferRef
    {
        T* data_;
        std::size_t count_;
        std::optional<std::bitset<BUFFER_COUNT>::reference> dirtyFlag_;

    public:
        BufferRef( T* data, std::size_t count, std::optional<std::bitset<BUFFER_COUNT>::reference> dirtyFlag ) noexcept;
        BufferRef( BufferRef<T>&& other ) noexcept;
        BufferRef( const BufferRef<T>& ) = delete;
        ~BufferRef() { if ( dirtyFlag_ ) *dirtyFlag_ = false; }

        T& operator []( std::size_t i ) const noexcept { return data_[i]; }
        T* data() const noexcept { return data_; };
        [[nodiscard]] std::size_t size() const noexcept { return data_ ? count_ : 0; }
        [[nodiscard]] std::size_t count() const noexcept { return count_; }
        [[nodiscard]] bool dirty() const noexcept { return bool( dirtyFlag_ ); }
    };

    template <BufferType type>
    BufferRef<typename ElementType<type>::type> prepareBuffer_( std::size_t elementCount ) const;

    template <BufferType type>
    BufferRef<typename ElementType<type>::type> loadBuffer_() const;

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

    template <BufferType type>
    void renderEdges_( const RenderParams& parameters, GLuint vao, GLuint vbo, const Color& color ) const;

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
    // Marks vertex normals' source
    mutable bool hasVertNormals_{ false };
};

}