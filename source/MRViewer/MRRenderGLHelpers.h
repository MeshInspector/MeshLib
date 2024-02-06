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

MRVIEWER_API void setTextureFilterType( FilterType filter, bool dim3d = false );
MRVIEWER_API void setTextureWrapType( WrapType wrap, bool dim3d = false );

// represents OpenGL 2D texture owner, and allows uploading data in it remembering texture size
class GlTexture2
{
public:
    constexpr static GLuint NO_TEX = 0;

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

// return real GL value for DepthFunction
// default is less
inline int getDepthFunctionLess( DepthFunction funcType )
{
    switch ( funcType )
    {
    case DepthFunction::Default:
    case DepthFunction::Less:
        return GL_LESS;
    case DepthFunction::Never:
        return GL_NEVER;
    case DepthFunction::LessOrEqual:
        return GL_LEQUAL;
    case DepthFunction::Equal:
        return GL_EQUAL;
    case DepthFunction::GreaterOrEqual:
        return GL_GEQUAL;
    case DepthFunction::Greater:
        return GL_GREATER;
    case DepthFunction::Always:
        return GL_ALWAYS;
    case DepthFunction::NotEqual:
        return GL_NOTEQUAL;
    default:
        return 0;
    }
}

// return real GL value for DepthFunction
// default is less or equal
inline int getDepthFunctionLEqual( DepthFunction funcType )
{
    if ( funcType == DepthFunction::Default )
        return GL_LEQUAL;
    return getDepthFunctionLess( funcType );
}

// class for easier rendering in framebuffer texture
class MRVIEWER_CLASS FramebufferData
{
public:
    // generates framebuffer and associated data
    // msaaPow - 2^msaaPow samples, msaaPow < 0 - use same default amount of samples
    // to resize: del(); gen( newSize, msaaPow );
    MRVIEWER_API void gen( const Vector2i& size, int msaaPow );
    // binds this framebuffer as main rendering target
    // clears it if `clear` flag is set
    MRVIEWER_API void bind( bool clear = true );
    // binds default framebuffer (and read/draw framebuffers)
    // make sure to bind correct framebuffer `getViewerInstance().bindSceneTexture( true )`
    MRVIEWER_API void bindDefault();
    // marks the texture to reading
    MRVIEWER_API void bindTexture();
    // copies picture rendered in this framebuffer to associated texutre for further use
    // and binds default framebuffer (and read/draw framebuffers)
    // make sure to bind correct framebuffer afterwards
    MRVIEWER_API void copyTextureBindDef();
    // removes this framebuffer
    MRVIEWER_API void del();
    // gets texture id for binding in other shaders
    unsigned getTexture() const { return resTexture_.getId(); }

    const Vector2i& getSize() const { return size_; }
private:
    void resize_( const Vector2i& size, int msaaPow );

    unsigned mainFramebuffer_{ 0 };
    unsigned colorRenderbuffer_{ 0 };
    unsigned depthRenderbuffer_{ 0 };
    unsigned copyFramebuffer_{ 0 };
    GlTexture2 resTexture_;
    Vector2i size_;
};

// class for rendering simple texture
class MRVIEWER_CLASS QuadTextureVertexObject
{
public:
    // generates simple quad for rendering
    MRVIEWER_API void gen();
    // binds simple quad vertex data
    MRVIEWER_API void bind();
    // removes this object
    MRVIEWER_API void del();
private:
    unsigned vao_;
    unsigned vbo_;
};

} //namespace MR
