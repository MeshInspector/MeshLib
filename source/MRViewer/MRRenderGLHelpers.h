#pragma once
#include "MRMesh/MRIRenderObject.h"
#include "MRGladGlfw.h"
#include "exports.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRVector2.h"
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

    auto getId() const { return bufferID_; }
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

// represents OpenGL 2D texture owner, and allows uploading data in it remembering texture size
class GlTexture2
{
    constexpr static GLuint NO_TEX = 0;
public:
    GlTexture2() = default;
    GlTexture2( const GlTexture2 & ) = delete;
    GlTexture2( GlTexture2 && r ) : textureID_( r.textureID_ ), size_( r.size_ ) { r.detach_(); }
    ~GlTexture2() { del(); }

    GlTexture2& operator =( const GlTexture2 & ) = delete;
    GlTexture2& operator =( GlTexture2 && r ) { del(); textureID_ = r.textureID_; size_ = r.size_; r.detach_(); return * this; }

    auto getId() const { return textureID_; }
    bool valid() const { return textureID_ != NO_TEX; }
    size_t size() const { return size_; }

    // generates new texture
    MRVIEWER_API void gen();

    // deletes the texture
    MRVIEWER_API void del();

    // binds current texture to OpenGL context
    MRVIEWER_API void bind();

    struct Settings
    {
        Vector2i resolution;
        size_t size() const { return size_t( resolution.x ) * resolution.y; }

        GLint internalFormat = GL_RGBA;
        GLint format = GL_RGBA;
        GLint type = GL_UNSIGNED_BYTE;
        WrapType wrap = WrapType::Mirror;
        FilterType filter = FilterType::Discrete;
    };

    // creates GL data texture using given data and binds it
    MRVIEWER_API void loadData( const Settings & settings, const char * arr );
    template<typename C>
    void loadData( const Settings & settings, const C & cont ) {
        assert( cont.size() >= settings.size() );
        loadData( settings, (const char *)cont.data() ); 
    }

    // binds current texture to OpenGL context, optionally refreshing its data
    MRVIEWER_API void loadDataOpt( bool refresh, const Settings & settings, const char * arr );
    template<typename C>
    void loadDataOpt( bool refresh, const Settings & settings, const C & cont ) {
        assert( !refresh || cont.size() >= settings.size() );
        loadDataOpt( refresh, settings, (const char *)cont.data() );
    }

private:
    /// another object takes control over the GL texture
    void detach_() { textureID_ = NO_TEX; size_ = 0; }

private:
    GLuint textureID_ = NO_TEX;
    size_t size_ = 0;
};

// represents OpenGL 3D texture owner, and allows uploading data in it remembering texture size
class GlTexture3
{
    constexpr static GLuint NO_TEX = 0;
public:
    GlTexture3() = default;
    GlTexture3( const GlTexture3& ) = delete;
    GlTexture3( GlTexture3&& r ) : textureID_( r.textureID_ ), size_( r.size_ ) { r.detach_(); }
    ~GlTexture3() { del(); }

    GlTexture3& operator =( const GlTexture3& ) = delete;
    GlTexture3& operator =( GlTexture3&& r ) { del(); textureID_ = r.textureID_; size_ = r.size_; r.detach_(); return * this; }

    auto getId() const { return textureID_; }
    bool valid() const { return textureID_ != NO_TEX; }
    size_t size() const { return size_; }

    // generates new texture
    MRVIEWER_API void gen();

    // deletes the texture
    MRVIEWER_API void del();

    // binds current texture to OpenGL context
    MRVIEWER_API void bind();

    struct Settings
    {
        Vector3i resolution;
        size_t size() const { return size_t( resolution.x ) * resolution.y * resolution.z; }

        GLint internalFormat = GL_RGBA;
        GLint format = GL_RGBA;
        GLint type = GL_UNSIGNED_BYTE;
        WrapType wrap = WrapType::Mirror;
        FilterType filter = FilterType::Discrete;
    };

    // creates GL data texture using given data and binds it
    MRVIEWER_API void loadData( const Settings & settings, const char * arr );
    template<typename C>
    void loadData( const Settings & settings, const C & cont ) {
        assert( cont.size() >= settings.size() );
        loadData( settings, (const char *)cont.data() ); 
    }

    // binds current texture to OpenGL context, optionally refreshing its data
    MRVIEWER_API void loadDataOpt( bool refresh, const Settings & settings, const char * arr );
    template<typename C>
    void loadDataOpt( bool refresh, const Settings & settings, const C & cont ) {
        assert( !refresh || cont.size() >= settings.size() );
        loadDataOpt( refresh, settings, (const char *)cont.data() );
    }

private:
    /// another object takes control over the GL texture
    void detach_() { textureID_ = NO_TEX; size_ = 0; }

private:
    GLuint textureID_ = NO_TEX;
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

// return real GL value for DepthFuncion
// default is less
inline int getDepthFunctionLess( DepthFuncion funcType )
{
    switch ( funcType )
    {
    case DepthFuncion::Default:
    case DepthFuncion::Less:
        return GL_LESS;
    case DepthFuncion::Never:
        return GL_NEVER;
    case DepthFuncion::LessOrEqual:
        return GL_LEQUAL;
    case DepthFuncion::Equal:
        return GL_EQUAL;
    case DepthFuncion::GreaterOrEqual:
        return GL_GEQUAL;
    case DepthFuncion::Greater:
        return GL_GREATER;
    case DepthFuncion::Always:
        return GL_ALWAYS;
    case DepthFuncion::NotEqual:
        return GL_NOTEQUAL;
    default:
        return 0;
    }
}

// return real GL value for DepthFuncion
// default is less or equal
inline int getDepthFunctionLEqual( DepthFuncion funcType )
{
    if ( funcType == DepthFuncion::Default )
        return GL_LEQUAL;
    return getDepthFunctionLess( funcType );
}

} //namespace MR
