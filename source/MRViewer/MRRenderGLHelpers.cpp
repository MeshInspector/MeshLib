#include "MRRenderGLHelpers.h"
#include "MRViewer.h"
#include "MRGLMacro.h"

namespace MR
{

void GlBuffer::gen()
{
    del();
    GL_EXEC( glGenBuffers( 1, &bufferID_ ) );
    assert( valid() );
}

void GlBuffer::del()
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

void GlBuffer::bind( GLenum target )
{ 
    assert( valid() );
    GL_EXEC( glBindBuffer( target, bufferID_ ) );
}

void GlBuffer::loadData( GLenum target, const char * arr, size_t arrSize )
{
    if ( !valid() )
        gen();
    bind( target );
    GLint64 bufSize = arrSize;
    auto maxUploadSize = ( GLint64( 1 ) << 32 ) - 4096; //4Gb - 4096, 4Gb is already too much
    if ( bufSize <= maxUploadSize )
    {
        // buffers less than 4Gb are ok to load immediately
        GL_EXEC( glBufferData( target, bufSize, arr, GL_DYNAMIC_DRAW ) );
    }
    else
    {
        // buffers more than 4Gb are better to split on chunks to avoid strange errors from GL or drivers
        GL_EXEC( glBufferData( target, bufSize, nullptr, GL_DYNAMIC_DRAW ) );
        GLint64 remStart = 0;
        auto remSize = bufSize;
        for ( ; remSize > maxUploadSize; remSize -= maxUploadSize, remStart += maxUploadSize )
        {
            GL_EXEC( glBufferSubData( target, remStart, maxUploadSize, arr + remStart ) );
        }
        GL_EXEC( glBufferSubData( target, remStart, remSize, arr + remStart ) );
    }
    size_ = arrSize;
}

void GlBuffer::loadDataOpt( GLenum target, bool refresh, const char * arr, size_t arrSize )
{
    if ( refresh )
        loadData( target, arr, arrSize );
    else
        bind( target );
}

void GlTexture2::gen()
{
    del();
    GL_EXEC( glGenTextures( 1, &textureID_ ) );
    assert( valid() );
}

void GlTexture2::del()
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

void GlTexture2::bind()
{ 
    assert( valid() );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, textureID_ ) );
}

void GlTexture2::loadData( const Settings & settings, const char * arr )
{
    if ( !valid() )
        gen();
    bind();

    GLint wrap = GL_MIRRORED_REPEAT;
    switch ( settings.wrap )
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
    GLint filter = settings.filter == FilterType::Linear ? GL_LINEAR : GL_NEAREST;

    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter ) );
    GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, settings.internalFormat, settings.resolution.x, settings.resolution.y, 0, settings.format, settings.type, arr ) );

    size_ = settings.size();
}

void GlTexture2::loadDataOpt( bool refresh, const Settings & settings, const char * arr )
{
    if ( refresh )
        loadData( settings, arr );
    else
        bind();
}

void GlTexture3::gen()
{
    del();
    GL_EXEC( glGenTextures( 1, &textureID_ ) );
    assert( valid() );
}

void GlTexture3::del()
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

void GlTexture3::bind()
{ 
    assert( valid() );
    GL_EXEC( glBindTexture( GL_TEXTURE_3D, textureID_ ) );
}

void GlTexture3::loadData( const Settings & settings, const char * arr )
{
    if ( !valid() )
        gen();
    bind();

    GLint wrap = GL_MIRRORED_REPEAT;
    switch ( settings.wrap )
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
    GLint filter = settings.filter == FilterType::Linear ? GL_LINEAR : GL_NEAREST;

    GL_EXEC( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter ) );
    GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
    GL_EXEC( glTexImage3D( GL_TEXTURE_3D, 0, settings.internalFormat, settings.resolution.x, settings.resolution.y, settings.resolution.z, 0, settings.format, settings.type, arr ) );

    size_ = settings.size();
}

void GlTexture3::loadDataOpt( bool refresh, const Settings & settings, const char * arr )
{
    if ( refresh )
        loadData( settings, arr );
    else
        bind();
}

GLint bindVertexAttribArray( const BindVertexAttribArraySettings & settings )
{
    GL_EXEC( GLint id = glGetAttribLocation( settings.program_shader, settings.name ) );
    if ( id < 0 )
        return id;
    if ( settings.arrSize == 0 && !settings.forceUse )
    {
        GL_EXEC( glDisableVertexAttribArray( id ) );
        settings.buf.del();
        return id;
    }

    settings.buf.loadDataOpt( GL_ARRAY_BUFFER, settings.refresh, settings.arr, settings.arrSize );

    // GL_FLOAT is left here consciously 
    if ( settings.isColor )
    {
        GL_EXEC( glVertexAttribPointer( id, settings.baseTypeElementsNumber, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0 ) );
    }
    else
    {
        GL_EXEC( glVertexAttribPointer( id, settings.baseTypeElementsNumber, GL_FLOAT, GL_FALSE, 0, 0 ) );
    }

    GL_EXEC( glEnableVertexAttribArray( id ) );
    return id;
}

} //namespace MR
