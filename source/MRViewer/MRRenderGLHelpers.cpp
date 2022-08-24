#include "MRRenderGLHelpers.h"
#include "MRViewer.h"

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

} //namespace MR
