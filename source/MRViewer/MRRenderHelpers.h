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
template <typename T>
class RenderBufferRef
{
public:
    RenderBufferRef( T* data, std::size_t glSize, bool dirty )
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

private:
    T* data_;
    std::size_t glSize_;
    bool dirty_;
};

/// ...
class RenderObjectBuffer
{
public:
    template <typename T>
    RenderBufferRef<T> prepareBuffer( std::size_t glSize, bool dirty = true )
    {
        auto memSize = sizeof( T ) * glSize;
        if ( buffer_.size() < memSize )
            buffer_.resize( memSize );

        return { reinterpret_cast<T*>( buffer_.data() ), glSize, dirty };
    }

    size_t heapBytes() const { return buffer_.heapBytes(); }

private:
    Buffer<std::byte> buffer_;
};

}
