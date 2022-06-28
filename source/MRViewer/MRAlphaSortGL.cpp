#include "MRAlphaSortGL.h"
#include "MRGLMacro.h"
#include "MRShadersHolder.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

namespace
{
constexpr unsigned cAlphaSortStoragePixelCapacity = 12;
}

namespace MR
{

AlphaSortGL::~AlphaSortGL()
{
    free();
}

void AlphaSortGL::init()
{
    if ( inited_ )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;

    inited_ = true;

    GL_EXEC( glGenVertexArrays( 1, &transparency_quad_vao ) );
    GL_EXEC( glGenBuffers( 1, &transparency_quad_vbo ) );
    GL_EXEC( glGenBuffers( 1, &transparency_shared_shader_data_vbo ) );
    GL_EXEC( glGenBuffers( 1, &transparency_atomic_counter_vbo ) );
    GL_EXEC( glGenBuffers( 1, &transparency_static_clean_vbo ) );
    GL_EXEC( glGenTextures( 1, &transparency_heads_texture_vbo ) );
}

void AlphaSortGL::free()
{
    if ( !inited_ )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    inited_ = false;

    GL_EXEC( glDeleteVertexArrays( 1, &transparency_quad_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &transparency_quad_vbo ) );
    GL_EXEC( glDeleteTextures( 1, &transparency_heads_texture_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &transparency_shared_shader_data_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &transparency_atomic_counter_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &transparency_static_clean_vbo ) );

}

void AlphaSortGL::clearTransparencyTextures() const
{
    if ( !inited_ )
        return;
    GL_EXEC( glBindBuffer( GL_SHADER_STORAGE_BUFFER, transparency_shared_shader_data_vbo ) );
    GL_EXEC( glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, transparency_shared_shader_data_vbo ) );
    GL_EXEC( glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 ) ); // unbind

    constexpr GLuint zero = 0;
    GL_EXEC( glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, transparency_atomic_counter_vbo ) );
    GL_EXEC( glBufferData( GL_ATOMIC_COUNTER_BUFFER, sizeof( GLuint ), &zero, GL_DYNAMIC_DRAW ) );
    GL_EXEC( glBindBufferBase( GL_ATOMIC_COUNTER_BUFFER, 0, transparency_atomic_counter_vbo ) );
    GL_EXEC( glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, 0 ) ); // unbind

    GL_EXEC( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, transparency_static_clean_vbo ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, transparency_heads_texture_vbo ) );
    GL_EXEC( glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0,
                              GLuint( width_ ), GLuint( height_ ),
                              GL_RED_INTEGER, GL_UNSIGNED_INT, NULL ) );
    GL_EXEC( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) ); // unbind
}

void AlphaSortGL::drawTransparencyTextureToScreen() const
{
    if ( !inited_ )
        return;
    GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    GL_EXEC( glDepthMask( GL_TRUE ) );
    GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );

    constexpr GLfloat transparencyTextureQuad[18] =
    {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f
    };

    GL_EXEC( glViewport( (GLsizei) 0, (GLsizei) 0, (GLsizei) width_, (GLsizei) height_ ) );

    // Send lines data to GL, install lines properties 
    GL_EXEC( glBindVertexArray( transparency_quad_vao ) );

    auto shader = ShadersHolder::getShaderId( ShadersHolder::TransparencyOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, transparency_quad_vbo ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, transparencyTextureQuad, GL_DYNAMIC_DRAW ) );

    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );

    GL_EXEC( glBindVertexArray( transparency_quad_vao ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );

    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, static_cast <GLsizei> ( 6 ) ) );
    GL_EXEC( glEnable( GL_DEPTH_TEST ) );
}

void AlphaSortGL::updateTransparencyTexturesSize( int width, int height )
{
    if ( transparency_heads_texture_vbo == 0 )
        return;

    if ( width == 0 || height == 0 )
        return;

    GL_EXEC( glDeleteTextures( 1, &transparency_heads_texture_vbo ) );
    GL_EXEC( glGenTextures( 1, &transparency_heads_texture_vbo ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, transparency_heads_texture_vbo ) );
    GL_EXEC( glTexStorage2D( GL_TEXTURE_2D, 1, GL_R32UI, GLuint( width ), GLuint( height ) ) );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glBindImageTexture( 0, transparency_heads_texture_vbo, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI ) );
#endif

    GLuint maxNodes = cAlphaSortStoragePixelCapacity * GLuint( width ) * GLuint( height );
    GLint  nodeSize = 5 * sizeof( GLfloat ) + sizeof( GLuint );

    GL_EXEC( glBindBuffer( GL_SHADER_STORAGE_BUFFER, transparency_shared_shader_data_vbo ) );
    GL_EXEC( glBufferData( GL_SHADER_STORAGE_BUFFER, maxNodes * nodeSize, NULL, GL_DYNAMIC_DRAW ) );
    GL_EXEC( glBindBuffer( GL_SHADER_STORAGE_BUFFER, 0 ) ); // unbind

    std::vector<GLuint> ones( GLuint( width ) * GLuint( height ), 0xFFFFFFFF );
    GL_EXEC( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, transparency_static_clean_vbo ) );
    GL_EXEC( glBufferData( GL_PIXEL_UNPACK_BUFFER, ones.size() * sizeof( GLuint ), ones.data(), GL_STATIC_COPY ) );
    GL_EXEC( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) ); // unbind

    width_ = width;
    height_ = height;
    clearTransparencyTextures();
}

}