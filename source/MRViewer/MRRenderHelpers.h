#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

// calc texture resolution, to fit MAX_TEXTURE_SIZE, and have minimal empty pixels
Vector2i calcTextureRes( int bufferSize, int maxTextWidth );

// provides access to shared buffer with type casting
template <typename T>
class RenderBufferRef
{
public:
    RenderBufferRef() = default;

    RenderBufferRef( T* data, std::size_t glSize, bool dirty )
        : data_( data )
        , glSize_( glSize )
        , dirty_( dirty )
    {
        if ( data_ == nullptr )
            assert( !dirty_ );
    }

    // returns reference to i-th element
    T& operator []( std::size_t i ) const noexcept
    {
        assert( dirty_ );
        return data_[i];
    }
    // returns pointer to buffer data
    T* data() const noexcept
    {
        return data_;
    }
    // returns actual buffer size
    [[nodiscard]] std::size_t size() const noexcept
    {
        return dirty_ ? glSize_ : 0;
    }
    // returns number of elements that are about to be loaded or already loaded to GL memory
    [[nodiscard]] std::size_t glSize() const noexcept
    {
        return glSize_;
    }
    // returns true if associated data were updated
    [[nodiscard]] bool dirty() const noexcept
    {
        return dirty_;
    }

private:
    T* data_{ nullptr };
    std::size_t glSize_{ 0 };
    bool dirty_{ false };
};

// provides shared buffer for loading different types of data to GL memory
class RenderObjectBuffer
{
    friend class GLStaticHolder;
    RenderObjectBuffer() = default;

public:
    template <typename T>
    RenderBufferRef<T> prepareBuffer( std::size_t glSize, bool dirty = true )
    {
        if ( dirty )
        {
            auto memSize = sizeof( T ) * glSize;
            if ( buffer_.size() < memSize )
                buffer_.resize( memSize );
        }
        return { reinterpret_cast<T*>( buffer_.data() ), glSize, dirty };
    }

    size_t heapBytes() const { return buffer_.heapBytes(); }

private:
    Buffer<std::byte> buffer_;
};

}
