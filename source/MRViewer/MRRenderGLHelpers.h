#pragma once

#include "MRGladGlfw.h"
#include "MRGLMacro.h"
#include "MRViewer.h"
#include <cassert>

namespace MR
{

// represents OpenGL buffer owner, and allows uploading data in it remembering buffer size
class GlBuffer
{
    constexpr static GLuint NO_BUF = 0;
public:
    GlBuffer() = default;
    GlBuffer( const GlBuffer & ) = delete;
    GlBuffer( GlBuffer && r ) : bufferID_( r.bufferID_ ), size_( r.size_ ) { r.detach_(); }
    ~GlBuffer() { del(); }

    GlBuffer& operator =( const GlBuffer & ) = delete;
    GlBuffer& operator =( GlBuffer && r ) { del(); bufferID_ = r.bufferID_; size_ = r.size_; r.detach_(); return * this; }

    bool valid() const { return bufferID_ != NO_BUF; }
    size_t size() const { return size_; }

    // generates new buffer
    void gen()
    {
        del();
        GL_EXEC( glGenBuffers( 1, &bufferID_ ) );
        assert( valid() );
    }

    // deletes the buffer
    void del()
    {
        if ( !valid() )
            return;
        if ( Viewer::constInstance()->isGLInitialized() && loadGL() )
        {
            GL_EXEC( glDeleteBuffers( 1, &bufferID_ ) );
        }
        bufferID_ = NO_BUF;
        size_ = 0;
    }

    // binds current buffer to OpenGL context
    void bind() { assert( valid() ); GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, bufferID_ ) ); }

    // creates GL data buffer using given data
    template<typename T>
    void loadData( const T * arr, size_t arrSize );

private:
    /// another object takes control over the GL buffer
    void detach_() { bufferID_ = NO_BUF; size_ = 0; }

private:
    GLuint bufferID_ = NO_BUF;
    size_t size_ = 0;
};

template<typename T>
void GlBuffer::loadData( const T * arr, size_t arrSize )
{
    if ( !valid() )
        gen();
    bind();
    GLint64 bufSize = sizeof( T ) * arrSize;
    auto maxUploadSize = ( GLint64( 1 ) << 32 ) - 4096; //4Gb - 4096, 4Gb is already too much
    if ( bufSize <= maxUploadSize )
    {
        // buffers less than 4Gb are ok to load immediately
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, bufSize, arr, GL_DYNAMIC_DRAW ) );
    }
    else
    {
        // buffers more than 4Gb are better to split on chunks to avoid strange errors from GL or drivers
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, bufSize, nullptr, GL_DYNAMIC_DRAW ) );
        GLint64 remStart = 0;
        auto remSize = bufSize;
        for ( ; remSize > maxUploadSize; remSize -= maxUploadSize, remStart += maxUploadSize )
        {
            GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, maxUploadSize, (const char *)arr + remStart ) );
        }
        GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, remSize, (const char *)arr + remStart ) );
    }
    size_ = size_t( bufSize );
}

template<typename T, template<typename, typename...> class C, typename... args>
GLint bindVertexAttribArray(
    const GLuint program_shader,
    const std::string& name,
    GlBuffer & buf,
    const C<T, args...>& V,
    int baseTypeElementsNumber,
    bool refresh,
    bool forceUse = false )
{
    GL_EXEC( GLint id = glGetAttribLocation( program_shader, name.c_str() ) );
    if ( id < 0 )
        return id;
    if ( V.size() == 0 && !forceUse )
    {
        GL_EXEC( glDisableVertexAttribArray( id ) );
        buf.del();
        return id;
    }

    if ( refresh )
        buf.loadData( V.data(), V.size() );
    else
        buf.bind();

    // GL_FLOAT is left here consciously 
    if constexpr ( std::is_same_v<Color, T> )
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0 ) );
    }
    else
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_FLOAT, GL_FALSE, 0, 0 ) );
    }

    GL_EXEC( glEnableVertexAttribArray( id ) );
    return id;
}
}