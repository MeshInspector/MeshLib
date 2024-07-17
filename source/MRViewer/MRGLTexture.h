#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"

#include "MRGladGlfw.h"
#include "MRMesh/MRVector3.h"

namespace MR
{

MRVIEWER_API void setTextureWrapType( WrapType wrapType, GLenum type = GL_TEXTURE_2D );
MRVIEWER_API void setTextureFilterType( FilterType filterType, GLenum type = GL_TEXTURE_2D );

// represents OpenGL 3D texture owner, and allows uploading data in it remembering texture size
class GlTexture
{
public:
    constexpr static GLuint NO_TEX = 0;

    GlTexture( GLenum  val );

    GlTexture( const GlTexture& ) = delete;
    GlTexture( GlTexture&& r );
    virtual ~GlTexture();

    GlTexture& operator =( const GlTexture& ) = delete;
    GlTexture& operator =( GlTexture&& r )
    {
        del(); textureID_ = r.textureID_; size_ = r.size_; r.detach_(); return *this;
    }

    auto getId() const
    {
        return textureID_;
    }

    bool valid() const;
    size_t size() const;

    // generates new texture
    MRVIEWER_API void gen();

    // deletes the texture
    MRVIEWER_API void del();

    // binds current texture to OpenGL context
    MRVIEWER_API void bind();

protected:

    GLuint textureID_ = NO_TEX;
    size_t size_ = 0;
    GLenum type_ = NO_TEX;

private:

    /// another object takes control over the GL texture
    void detach_();
};
}
