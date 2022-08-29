#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

// calc texture resolution, to fit MAX_TEXTURE_SIZE, and have minimal empty pixels
Vector2i calcTextureRes( int bufferSize, int maxTextWidth );

/// ...
using RenderDirtyFlag = std::underlying_type_t<DirtyFlags>;

/// ...
template <RenderDirtyFlag dirtyFlag>
struct RenderBufferType;

/// ...
template <RenderDirtyFlag dirtyFlag>
class RenderBufferRef
{
public:
    using DataType = typename RenderBufferType<dirtyFlag>::type;

    RenderBufferRef( DataType* data, std::size_t glSize, bool dirty )
        : data_( data )
        , glSize_( glSize )
        , dirty_( dirty )
    {
        if ( data_ == nullptr )
            assert( !dirty_ );
    }

    /// ...
    DataType& operator []( std::size_t i ) const noexcept
    {
        assert( dirty_ );
        return data_[i];
    }
    /// ...
    DataType* data() const noexcept
    {
        return data_;
    }
    /// returns actual buffer size
    [[nodiscard]] std::size_t size() const noexcept
    {
        return dirty_ ? glSize_ : 0;
    }
    /// returns number of elements that are about to be loaded or already loaded to GL memory
    [[nodiscard]] std::size_t glSize() const noexcept
    {
        return glSize_;
    }
    /// ...
    [[nodiscard]] bool dirty() const noexcept
    {
        return dirty_;
    }

private:
    DataType* data_;
    std::size_t glSize_;
    bool dirty_;
};

/// ...
class RenderObjectBuffer
{
public:
    template <RenderDirtyFlag dirtyFlag>
    RenderBufferRef<dirtyFlag> prepareBuffer( std::size_t glSize, bool dirty = true )
    {
        using DataType = typename RenderBufferRef<dirtyFlag>::DataType;
        auto memSize = sizeof( DataType ) * glSize;
        if ( buffer_.size() < memSize )
            buffer_.resize( memSize );

        return { reinterpret_cast<DataType*>( buffer_.data() ), glSize, dirty };
    }

    size_t heapBytes() const { return buffer_.heapBytes(); }

private:
    Buffer<std::byte> buffer_;
};

/// ...
constexpr std::underlying_type_t<DirtyFlags> DIRTY_EDGE = 0x40000;
static_assert( DIRTY_EDGE == DIRTY_ALL + 1 );

template<> struct RenderBufferType<DIRTY_POSITION> { using type = Vector3f; };
template<> struct RenderBufferType<DIRTY_UV> { using type = UVCoord; };
template<> struct RenderBufferType<DIRTY_VERTS_RENDER_NORMAL> { using type = Vector3f; };
template<> struct RenderBufferType<DIRTY_FACES_RENDER_NORMAL> { using type = Vector4f; };
template<> struct RenderBufferType<DIRTY_CORNERS_RENDER_NORMAL> { using type = Vector3f; };
template<> struct RenderBufferType<DIRTY_SELECTION> { using type = unsigned; };
template<> struct RenderBufferType<DIRTY_TEXTURE> { using type = Color; };
template<> struct RenderBufferType<DIRTY_FACE> { using type = Vector3i; };
template<> struct RenderBufferType<DIRTY_VERTS_COLORMAP> { using type = Color; };
template<> struct RenderBufferType<DIRTY_PRIMITIVE_COLORMAP> { using type = Color; };
template<> struct RenderBufferType<DIRTY_BORDER_LINES> { using type = Vector3f; };
template<> struct RenderBufferType<DIRTY_EDGES_SELECTION> { using type = Vector3f; };
template<> struct RenderBufferType<DIRTY_EDGE> { using type = Vector2i; };

}
