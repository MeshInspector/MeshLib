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

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;
    virtual size_t heapBytes() const override;

private:
    const ObjectMeshHolder* objMesh_;

    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    mutable Buffer<std::byte> bufferObj_;

    using DirtyFlag = uint32_t;

    template <DirtyFlag>
    struct BufferTypeHelper;
    template <DirtyFlag dirtyFlag>
    using BufferType = typename BufferTypeHelper<dirtyFlag>::type;

    mutable std::array<std::size_t, 8 * sizeof( DirtyFlag )> bufferGLSize_;
    template <DirtyFlag>
    std::size_t& getGLSize_() const;

    template <typename T>
    class BufferRef
    {
        T* data_;
        std::size_t glSize_;
        DirtyFlag* dirtyMask_;
        DirtyFlag dirtyFlag_;

    public:
        BufferRef( T* data, std::size_t glSize, DirtyFlag* dirtyMask, DirtyFlag dirtyFlag ) noexcept;
        BufferRef( BufferRef<T>&& other ) noexcept;
        BufferRef( const BufferRef<T>& ) = delete;
        ~BufferRef() { if ( dirtyMask_ ) *dirtyMask_ ^= dirtyFlag_; }

        T& operator []( std::size_t i ) const noexcept { return data_[i]; }
        T* data() const noexcept { return data_; };
        [[nodiscard]] std::size_t size() const noexcept { return data_ ? glSize_ : 0; }
        [[nodiscard]] std::size_t glSize() const noexcept { return glSize_; }
        [[nodiscard]] bool dirty() const noexcept { return dirtyMask_ && ( *dirtyMask_ & dirtyFlag_ ); }
    };

    template <DirtyFlag dirtyFlag>
    BufferRef<BufferType<dirtyFlag>> prepareBuffer_( std::size_t glSize ) const;

    template <DirtyFlag dirtyFlag>
    BufferRef<BufferType<dirtyFlag>> loadBuffer_() const;

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

    mutable GlBuffer facesIndicesBuffer_;
    mutable GlBuffer edgesIndicesBuffer_;
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    int maxTexSize_{ 0 };

    template <DirtyFlag>
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
    mutable DirtyFlag dirty_;
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    mutable bool normalsBound_{ false };
};

}