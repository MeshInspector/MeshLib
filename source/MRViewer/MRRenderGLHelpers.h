#pragma once

#include "MRGladGlfw.h"
#include "exports.h"
#include "MRMesh/MRColor.h"
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
    MRVIEWER_API void gen();

    // deletes the buffer
    MRVIEWER_API void del();

    // binds current buffer to OpenGL context
    MRVIEWER_API void bind( GLenum target );

    // creates GL data buffer using given data and binds it
    MRVIEWER_API void loadData( GLenum target, const char * arr, size_t arrSize );
    template<typename T>
    void loadData( GLenum target, const T * arr, size_t arrSize ) { loadData( target, (const char *)arr, sizeof( T ) * arrSize ); }
    template<typename C>
    void loadData( GLenum target, const C & cont ) { loadData( target, cont.data(), cont.size() ); }

    // binds current buffer to OpenGL context, optionally refreshing its data
    MRVIEWER_API void loadDataOpt( GLenum target, bool refresh, const char * arr, size_t arrSize );
    template<typename T>
    void loadDataOpt( GLenum target, bool refresh, const T * arr, size_t arrSize ) { loadDataOpt( target, refresh, (const char *)arr, sizeof( T ) * arrSize ); }
    template<typename C>
    void loadDataOpt( GLenum target, bool refresh, const C & cont ) { loadDataOpt( target, refresh, cont.data(), cont.size() ); }

private:
    /// another object takes control over the GL buffer
    void detach_() { bufferID_ = NO_BUF; size_ = 0; }

private:
    GLuint bufferID_ = NO_BUF;
    size_t size_ = 0;
};

struct BindVertexAttribArraySettings
{
    GLuint program_shader = 0;
    const char * name = nullptr;
    GlBuffer & buf;
    const char * arr = nullptr;
    size_t arrSize = 0;
    int baseTypeElementsNumber = 0;
    bool refresh = false;
    bool forceUse = false;
    bool isColor = false;
};

MRVIEWER_API GLint bindVertexAttribArray( const BindVertexAttribArraySettings & settings );

template<typename T, template<typename, typename...> class C, typename... args>
inline GLint bindVertexAttribArray(
    const GLuint program_shader,
    const char * name,
    GlBuffer & buf,
    const C<T, args...>& V,
    int baseTypeElementsNumber,
    bool refresh,
    bool forceUse = false )
{
    BindVertexAttribArraySettings settings =
    {
        .program_shader = program_shader,
        .name = name,
        .buf = buf,
        .arr = (const char*)V.data(),
        .arrSize = sizeof(T) * V.size(),
        .baseTypeElementsNumber = baseTypeElementsNumber,
        .refresh = refresh,
        .forceUse = forceUse,
        .isColor = std::is_same_v<Color, T>
    };
    return bindVertexAttribArray( settings );
}

template <typename T, std::size_t N>
inline GLint bindVertexAttribArray(
    const GLuint program_shader,
    const char * name,
    GlBuffer & buf,
    const std::array<T, N>& V,
    int baseTypeElementsNumber,
    bool refresh,
    bool forceUse = false )
{
    BindVertexAttribArraySettings settings =
    {
        .program_shader = program_shader,
        .name = name,
        .buf = buf,
        .arr = (const char*)V.data(),
        .arrSize = sizeof(T) * N,
        .baseTypeElementsNumber = baseTypeElementsNumber,
        .refresh = refresh,
        .forceUse = forceUse,
        .isColor = std::is_same_v<Color, T>
    };
    return bindVertexAttribArray( settings );
}

} //namespace MR
