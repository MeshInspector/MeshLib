#include "MRRenderGLHelpers.h"
#include "MRViewer.h"
#include "MRGLStaticHolder.h"

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

void GlTexture2::texImage_( const Settings& settings, const char* arr )
{
    GL_EXEC( glTexImage2D( type_, 0, settings.internalFormat, settings.resolution.x, settings.resolution.y, 0, settings.format, settings.type, arr ) );
}

void GlTexture3::texImage_( const Settings& settings, const char* arr )
{
    GL_EXEC( glTexImage3D( type_, 0, settings.internalFormat, settings.resolution.x, settings.resolution.y, settings.resolution.z, 0, settings.format, settings.type, arr ) );
}

void GlTexture2DArray::texImage_( const Settings& settings, const char* arr )
{
    GL_EXEC( glTexImage3D( type_, 0, settings.internalFormat, settings.resolution.x, settings.resolution.y, settings.resolution.z, 0, settings.format, settings.type, arr ) );
}

GLint bindVertexAttribArray( const BindVertexAttribArraySettings & settings )
{
    GLint id = GL_EXEC( glGetAttribLocation( settings.program_shader, settings.name ) );
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

void bindDepthPeelingTextures( GLuint shaderId, const TransparencyMode& tMode, GLenum startGLTextureIndex )
{
    if ( !tMode.isDepthPeelingEnabled() )
        return;

    GL_EXEC( glActiveTexture( startGLTextureIndex ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, tMode.getBGDepthPeelingDepthTextureId() ) );
    setTextureWrapType( WrapType::Clamp );
    setTextureFilterType( FilterType::Linear );
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "dp_bg_depths" ), startGLTextureIndex - GL_TEXTURE0 ) );

    GL_EXEC( glActiveTexture( startGLTextureIndex + 1 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, tMode.getFGDepthPeelingColorTextureId() ) );
    setTextureWrapType( WrapType::Clamp );
    setTextureFilterType( FilterType::Discrete );
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "dp_fg_colors" ), startGLTextureIndex - GL_TEXTURE0 + 1 ) );

    GL_EXEC( glActiveTexture( startGLTextureIndex + 2 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, tMode.getFGDepthPeelingDepthTextureId() ) );
    setTextureWrapType( WrapType::Clamp );
    setTextureFilterType( FilterType::Discrete );
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "dp_fg_depths" ), startGLTextureIndex - GL_TEXTURE0 + 2 ) );
}

void FramebufferData::gen( const Vector2i& size, bool copyDepth, int msaaPow, bool highPrecisionDepth )
{
    highPrecisionDepth_ = highPrecisionDepth;
    // Create an initial multisampled framebuffer
    GL_EXEC( glGenFramebuffers( 1, &mainFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    // create a color renderbuffer
    GL_EXEC( glGenRenderbuffers( 1, &colorRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, colorRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );

    // create a renderbuffer object for depth attachments
    GL_EXEC( glGenRenderbuffers( 1, &depthRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, depthRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // configure second post-processing framebuffer
    GL_EXEC( glGenFramebuffers( 1, &copyFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, copyFramebuffer_ ) );
    // create a color attachment texture
    resColorTexture_.gen();
    if ( copyDepth )
        resDepthTexture_.gen();
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    resize_( size, msaaPow );
}

void FramebufferData::bind( bool clear, float clearDepth )
{
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    // Clear the buffer
    if ( clear )
    {
        constexpr float cClearValue[4] = { 0.0f,0.0f,0.0f,0.0f };
        GL_EXEC( glClearBufferfv( GL_COLOR, 0, cClearValue ) );
        GL_EXEC( glClearBufferfv( GL_DEPTH, 0, &clearDepth ) );
    }
    isBound_ = true;
}

void FramebufferData::bindDefault()
{
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    isBound_ = false;
}

void FramebufferData::bindTexture( bool color, bool depth )
{
    if ( color )
        resColorTexture_.bind();
    if ( depth && resDepthTexture_.valid() )
        resDepthTexture_.bind();
}

void FramebufferData::copyTextureBindDef()
{
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, mainFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, copyFramebuffer_ ) );
    auto copyBits = GL_COLOR_BUFFER_BIT;
    if ( resDepthTexture_.valid() )
        copyBits |= GL_DEPTH_BUFFER_BIT;
    GL_EXEC( glBlitFramebuffer( 0, 0, size_.x, size_.y, 0, 0, size_.x, size_.y, copyBits, GL_NEAREST ) );
    bindDefault();
}

void FramebufferData::del()
{
    resColorTexture_.del();
    if ( resDepthTexture_.valid() )
        resDepthTexture_.del();
    GL_EXEC( glDeleteFramebuffers( 1, &mainFramebuffer_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &copyFramebuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &depthRenderbuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &colorRenderbuffer_ ) );
}

void FramebufferData::draw( QuadTextureVertexObject& quadObject, const DrawParams& params ) const
{
#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    GL_EXEC( glViewport( 0, 0, params.size.x, params.size.y ) );

    bool useDepthTexture = !params.forceSimpleDepthDraw && resDepthTexture_.valid();
    auto shader = GLStaticHolder::getShaderId( useDepthTexture ? GLStaticHolder::DepthOverlayQuad : GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject.bind();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, getColorTexture() ) );
    setTextureWrapType( params.wrap );
    setTextureFilterType( params.filter );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );

    if ( useDepthTexture )
    {
        GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, getDepthTexture() ) );
        setTextureWrapType( params.wrap );
        setTextureFilterType( params.filter );
        GL_EXEC( glUniform1i( glGetUniformLocation( shader, "depths" ), 1 ) );
    }
    else
    {
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "depth" ), params.simpleDepth ) );
    }
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "viewportSize" ), float( params.size.x ), float( params.size.y ) ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );

#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
}

void FramebufferData::resize_( const Vector2i& size, int msaaPow )
{
    size_ = size;
    int samples = 0;
    if ( msaaPow < 0 )
    {
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &samples ) );
    }
    else
    {
        samples = 1 << msaaPow;
    }

    int maxSamples = 0;
    GL_EXEC( glGetIntegerv( GL_MAX_SAMPLES, &maxSamples ) );
    if ( maxSamples < 1 )
        maxSamples = 1;
    samples = std::clamp( samples, 1, maxSamples );

    bool multisample = samples > 1;


    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, colorRenderbuffer_ ) );
    if ( multisample )
    {
        GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_RGBA8, size.x, size.y ) );
    }
    else
    {
        GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, GL_RGBA8, size.x, size.y ) );
    }
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );

    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, depthRenderbuffer_ ) );
    auto depthComponent = highPrecisionDepth_ ? GL_DEPTH_COMPONENT32F : GL_DEPTH_COMPONENT24;
    if ( multisample )
    {
        GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, depthComponent, size.x, size.y ) );
    }
    else
    {
        GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, depthComponent, size.x, size.y ) );
    }
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, copyFramebuffer_ ) );

    resColorTexture_.loadData( {.resolution = Vector3i(size.x, size.y, 1), .wrap = WrapType::Clamp, .filter = FilterType::Linear }, ( const char* ) nullptr );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resColorTexture_.getId(), 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    if ( resDepthTexture_.valid() )
    {
        resDepthTexture_.loadData( { 
            .resolution = Vector3i( size.x, size.y, 1 ), 
            .internalFormat = depthComponent,
            .format = GL_DEPTH_COMPONENT, 
            .type = highPrecisionDepth_ ? GL_FLOAT : GL_UNSIGNED_INT,
            .wrap = WrapType::Clamp, 
            .filter = FilterType::Discrete }, ( const char* ) nullptr );
        GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, resDepthTexture_.getId(), 0 ) );
        assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    }
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void QuadTextureVertexObject::gen()
{
    constexpr GLfloat quad[18] =
    {
        -1.0f, -1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        1.0f,  1.0f, 0.99f
    };
    GL_EXEC( glGenVertexArrays( 1, &vao_ ) );
    GL_EXEC( glGenBuffers( 1, &vbo_ ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, quad, GL_STATIC_DRAW ) );
}

void QuadTextureVertexObject::bind()
{
    GL_EXEC( glBindVertexArray( vao_ ) );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) );
    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );
}

void QuadTextureVertexObject::del()
{
    GL_EXEC( glDeleteVertexArrays( 1, &vao_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vbo_ ) );
}

} //namespace MR
