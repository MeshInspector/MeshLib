#pragma once
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

/// ...
class RenderObjectBuffer
{
public:
    /// ...
    template <typename T>
    class BufferRef
    {
        T* data_;
        std::size_t glSize_;
        bool dirty_;

    public:
        BufferRef( T* data, std::size_t glSize, bool dirty )
            : data_( data )
            , glSize_( glSize )
            , dirty_( dirty )
        {
            if ( data_ == nullptr )
                assert( !dirty_ );
        }

        /// ...
        T& operator []( std::size_t i ) const noexcept
        {
            assert( dirty_ );
            return data_[i];
        }
        /// ...
        T* data() const noexcept
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
    };

    using DirtyFlag = uint32_t;

    /// ...
    template <DirtyFlag>
    struct BufferTypeHelper;
    template <DirtyFlag dirtyFlag>
    using BufferType = typename BufferTypeHelper<dirtyFlag>::type;

    template <DirtyFlag dirtyFlag>
    BufferRef<BufferType<dirtyFlag>> prepareBuffer( std::size_t glSize, bool dirty = true )
    {
        auto memSize = sizeof( BufferType<dirtyFlag> ) * glSize;
        if ( buffer_.size() < memSize )
            buffer_.resize( memSize );

        return { reinterpret_cast<BufferType<dirtyFlag>*>( buffer_.data() ), glSize, dirty };
    }

    size_t heapBytes() const { return buffer_.heapBytes(); }

private:
    Buffer<std::byte> buffer_;
};

/// ...
constexpr std::underlying_type_t<DirtyFlags> DIRTY_EDGE = 0x40000;
static_assert( DIRTY_EDGE == DIRTY_ALL + 1 );

template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_POSITION> { using type = Vector3f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_UV> { using type = UVCoord; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_VERTS_RENDER_NORMAL> { using type = Vector3f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_FACES_RENDER_NORMAL> { using type = Vector4f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_CORNERS_RENDER_NORMAL> { using type = Vector3f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_SELECTION> { using type = unsigned; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_TEXTURE> { using type = Color; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_FACE> { using type = Vector3i; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_VERTS_COLORMAP> { using type = Color; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_PRIMITIVE_COLORMAP> { using type = Color; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_BORDER_LINES> { using type = Vector3f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_EDGES_SELECTION> { using type = Vector3f; };
template<> struct RenderObjectBuffer::BufferTypeHelper<DIRTY_EDGE> { using type = Vector2i; };

} // namespace MR
