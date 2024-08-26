#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"

#include "MRGladGlfw.h"
#include "MRMesh/MRVector3.h"

namespace MR
{

MRVIEWER_API void setTextureWrapType( WrapType wrapType, GLenum type = GL_TEXTURE_2D );
MRVIEWER_API void setTextureFilterType( FilterType filterType, GLenum type = GL_TEXTURE_2D );

// represents OpenGL texture owner, and allows uploading data in it remembering texture size
class GlTexture
{
public:
    constexpr static GLuint NO_TEX = 0;

    MRVIEWER_API GlTexture( GLenum  val );

    GlTexture( const GlTexture& ) = delete;
    MRVIEWER_API GlTexture( GlTexture&& r );
    MRVIEWER_API virtual ~GlTexture();

    GlTexture& operator =( const GlTexture& ) = delete;
    GlTexture& operator =( GlTexture&& r )
    {
        del(); textureID_ = r.textureID_; size_ = r.size_; r.detach_(); return *this;
    }

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
        // the X and Y components are the dimensions of the target texture
        // Z - 1 for texture2d
        // Z - number of textures for texture2d array
        // Z - Z dimensions for texture3d
        Vector3i resolution;
        size_t size() const
        {
            return size_t( resolution.x ) * resolution.y * resolution.z;
        }

        GLint internalFormat = GL_RGBA;
        GLint format = GL_RGBA;
        GLint type = GL_UNSIGNED_BYTE;
        WrapType wrap = WrapType::Mirror;
        FilterType filter = FilterType::Discrete;
    };

    // creates GL data texture using given data and binds it
    MRVIEWER_API void loadData( const Settings& settings, const char* arr );
    template<typename C>
    void loadData( const Settings& settings, const C& cont )
    {
        assert( cont.size() >= settings.size() );
        loadData( settings, ( const char* )cont.data() );
    }

    // binds current texture to OpenGL context, optionally refreshing its data
    MRVIEWER_API void loadDataOpt( bool refresh, const Settings& settings, const char* arr );
    template<typename C>
    void loadDataOpt( bool refresh, const Settings& settings, const C& cont )
    {
        assert( !refresh || cont.size() >= settings.size() );
        loadDataOpt( refresh, settings, ( const char* )cont.data() );
    }

protected:
    virtual void texImage_( const Settings& settings, const char* arr ) = 0;
    GLuint textureID_ = NO_TEX;
    size_t size_ = 0;
    GLenum type_ = NO_TEX;

private:

    /// another object takes control over the GL texture
    void detach_();
};

} //namespace MR
