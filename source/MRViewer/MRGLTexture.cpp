#include "MRGLTexture.h"

#include "MRGLMacro.h"
#include "MRViewer.h"

namespace MR
{

void setTextureWrapType( WrapType wrapType, GLenum type )
{
    GLint wrap = GL_MIRRORED_REPEAT;
    switch ( wrapType )
    {
    default:
    case WrapType::Clamp:
        wrap = GL_CLAMP_TO_EDGE;
        break;
    case WrapType::Repeat:
        wrap = GL_REPEAT;
        break;
    case WrapType::Mirror:
        wrap = GL_MIRRORED_REPEAT;
        break;
    }

    GL_EXEC( glTexParameteri( type, GL_TEXTURE_WRAP_S, wrap ) );
    GL_EXEC( glTexParameteri( type, GL_TEXTURE_WRAP_T, wrap ) );
    if ( type == GL_TEXTURE_3D )
    {
        GL_EXEC( glTexParameteri( type, GL_TEXTURE_WRAP_R, wrap ) );
    }
}

void setTextureFilterType( FilterType filterType, GLenum type )
{
    GLint filter = filterType == FilterType::Linear ? GL_LINEAR : GL_NEAREST;
    GL_EXEC( glTexParameteri( type, GL_TEXTURE_MIN_FILTER, filter ) );
    GL_EXEC( glTexParameteri( type, GL_TEXTURE_MAG_FILTER, filter ) );
}

GlTexture::GlTexture( GLenum type )
{
    type_ = type;
}

GlTexture::GlTexture( GlTexture&& r ) : textureID_( r.textureID_ ), size_( r.size_ ), type_( r.type_ )
{
    r.detach_();
}
GlTexture::~GlTexture()
{
    del();
}

void GlTexture::gen()
{
    del();
    GL_EXEC( glGenTextures( 1, &textureID_ ) );
    assert( valid() );
}

void GlTexture::del()
{
    if ( !valid() )
        return;
    if ( Viewer::constInstance()->isGLInitialized() && loadGL() )
    {
        GL_EXEC( glDeleteTextures( 1, &textureID_ ) );
    }
    textureID_ = NO_TEX;
    size_ = 0;
}

void GlTexture::bind()
{
    assert( valid() );
    GL_EXEC( glBindTexture( type_, textureID_ ) );
}

void GlTexture::loadData( const Settings& settings, const char* arr )
{
    if ( !valid() )
        gen();
    bind();

    setTextureWrapType( settings.wrap, type_ );
    setTextureFilterType( settings.filter, type_ );
    GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
    texImage_( settings, arr );

    size_ = settings.size();
}

void GlTexture::loadDataOpt( bool refresh, const Settings& settings, const char* arr )
{
    if ( refresh )
        loadData( settings, arr );
    else
        bind();
}

void GlTexture::detach_()
{
    textureID_ = NO_TEX; size_ = 0;
}
}
