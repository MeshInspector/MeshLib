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

void GlBuffer::bind()
{ 
    assert( valid() );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, bufferID_ ) );
}

void GlBuffer::loadData( const char * arr, size_t arrSize )
{
    if ( !valid() )
        gen();
    bind();
    GLint64 bufSize = arrSize;
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
            GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, maxUploadSize, arr + remStart ) );
        }
        GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, remSize, arr + remStart ) );
    }
    size_ = arrSize;
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

    if ( settings.refresh )
        settings.buf.loadData( settings.arr, settings.arrSize );
    else
        settings.buf.bind();

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
