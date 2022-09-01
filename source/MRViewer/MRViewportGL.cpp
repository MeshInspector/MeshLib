#include "MRViewportGL.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRMatrix4.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

static_assert( sizeof( Vector3f ) == 3 * sizeof( float ), "wrong size of Vector3f" );
static_assert( sizeof( Vector4f ) == 4 * sizeof( float ), "wrong size of Vector4f" );
static_assert( sizeof( LineSegm3f ) == 6 * sizeof( float ), "wrong size of LineSegm3f" );
static_assert( sizeof( SegmEndColors ) == 8 * sizeof( float ), "wrong size of SegmEndColors" );

ViewportGL& ViewportGL::operator=( ViewportGL&& other ) noexcept
{
    free();

    add_line_colors_vbo = other.add_line_colors_vbo;
    add_line_vbo = other.add_line_vbo;
    add_line_vao = other.add_line_vao;

    add_point_colors_vbo = other.add_point_colors_vbo;
    add_point_vbo = other.add_point_vbo;
    add_point_vao = other.add_point_vao;

    border_line_vbo = other.border_line_vbo;
    border_line_vao = other.border_line_vao;

    previewLines_ = other.previewLines_;
    previewPoints_ = other.previewPoints_;

    lines_dirty = true;
    points_dirty = true;

    inited_ = other.inited_;

    other.inited_ = false;

    return *this;
}

ViewportGL::ViewportGL( ViewportGL&& other ) noexcept
{
    *this = std::move( other );
}

ViewportGL::~ViewportGL()
{
    free();
}

void ViewportGL::init()
{
    if ( inited_ )
        return;

    if ( !Viewer::constInstance()->isGLInitialized() )
        return;

    // lines 
    GL_EXEC( glGenVertexArrays( 1, &add_line_vao ) );
    GL_EXEC( glGenBuffers( 1, &add_line_vbo ) );
    GL_EXEC( glGenBuffers( 1, &add_line_colors_vbo ) );

    // points 
    GL_EXEC( glGenVertexArrays( 1, &add_point_vao ) );
    GL_EXEC( glGenBuffers( 1, &add_point_vbo ) );
    GL_EXEC( glGenBuffers( 1, &add_point_colors_vbo ) );

    // border 
    GL_EXEC( glGenVertexArrays( 1, &border_line_vao ) );
    GL_EXEC( glGenBuffers( 1, &border_line_vbo ) );

    inited_ = true;
}

void ViewportGL::free()
{
    if ( !inited_ )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;

    setLinesWithColors( { {},{} } );
    setPointsWithColors( { {},{} } );

    GL_EXEC( glDeleteVertexArrays( 1, &add_line_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &add_line_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &add_line_colors_vbo ) );

    GL_EXEC( glDeleteVertexArrays( 1, &add_point_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &add_point_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &add_point_colors_vbo ) );

    GL_EXEC( glDeleteVertexArrays( 1, &border_line_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &border_line_vbo ) );

    inited_ = false;
}

void ViewportGL::drawLines( const RenderParams& params ) const
{
    if ( previewLines_.lines.empty() )
        return;

    if ( !inited_ )
    {
        lines_dirty = false;
        return;
    }
    // set GL_DEPTH_TEST specified for lines
    if ( params.depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( (GLsizei) params.viewport.x, (GLsizei) params.viewport.y,
                         (GLsizei) params.viewport.z, (GLsizei) params.viewport.w ) );
    // Send lines data to GL, install lines properties 
    GL_EXEC( glBindVertexArray( add_line_vao ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::AdditionalLines );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrixPtr ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), params.zOffset * params.cameraZoom ) );

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, add_line_colors_vbo ) );
    if ( lines_dirty )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( SegmEndColors ) * previewLines_.colors.size(), previewLines_.colors.data(), GL_DYNAMIC_DRAW ) );
    }
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );


    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, add_line_vbo ) );
    if ( lines_dirty )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( LineSegm3f ) * previewLines_.lines.size(), previewLines_.lines.data(), GL_DYNAMIC_DRAW ) );
    }
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, previewLines_.lines.size() );

    GL_EXEC( glBindVertexArray( add_line_vao ) );
    GL_EXEC( glLineWidth( static_cast <GLfloat> ( params.width ) ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, static_cast <GLsizei> ( previewLines_.lines.size() * 2 ) ) );

    lines_dirty = false;
}

void ViewportGL::drawPoints( const RenderParams& params ) const
{
    if ( previewPoints_.points.empty() )
        return;

    if ( !inited_ )
    {
        points_dirty = false;
        return;
    }
    // set GL_DEPTH_TEST specified for points 
    if ( params.depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( (GLsizei) params.viewport.x, (GLsizei) params.viewport.y,
                         (GLsizei) params.viewport.z, (GLsizei) params.viewport.w ) );
    // Send points data to GL, install points properties 
    GL_EXEC( glBindVertexArray( add_point_vao ) );

    // AdditionalPointsNoOffset exists for old intel gpu (Intel HD 4000)
    auto shader = params.zOffset == 0.0f ? GLStaticHolder::getShaderId( GLStaticHolder::AdditionalPointsNoOffset ) : GLStaticHolder::getShaderId( GLStaticHolder::AdditionalPoints );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrixPtr ) );

    if ( params.zOffset != 0.0f )
    {
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), params.zOffset * params.cameraZoom ) );
    }

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, add_point_colors_vbo ) );
    if ( points_dirty )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( Vector4f ) * previewPoints_.colors.size(), previewPoints_.colors.data(), GL_DYNAMIC_DRAW ) );
    }
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, add_point_vbo ) );
    if ( points_dirty )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( Vector3f ) * previewPoints_.points.size(), previewPoints_.points.data(), GL_DYNAMIC_DRAW ) );
    }
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointArraySize, previewPoints_.points.size() );

    GL_EXEC( glBindVertexArray( add_point_vao ) );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), params.width ) );
#else
    GL_EXEC( glPointSize( params.width ) );
#endif
    GL_EXEC( glDrawArrays( GL_POINTS, 0, static_cast <GLsizei> ( previewPoints_.points.size() ) ) );
    points_dirty = false;
}

void ViewportGL::drawBorder( const BaseRenderParams& params, const Color& color ) const
{
    if ( !inited_ )
        return;
    constexpr GLfloat border[24] =
    {
        -1.f,-1.f,0.f,
        -1.f,1.f,0.f,
        -1.f,1.f,0.f,
        1.f,1.f,0.f,
        1.f,1.f,0.f,
        1.f,-1.f,0.f,
        1.f,-1.f,0.f,
        -1.f,-1.f,0.f,
    };

    GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    GL_EXEC( glViewport( (GLsizei) params.viewport.x, (GLsizei) params.viewport.y, 
                         (GLsizei) params.viewport.z, (GLsizei) params.viewport.w ) );

    // Send lines data to GL, install lines properties 
    GL_EXEC( glBindVertexArray( border_line_vao ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ViewportBorder );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( GLint colorid = glGetUniformLocation( shader, "user_color" ) );
    const auto borderColor = Color( Vector4f( color ) );
    GL_EXEC( glUniform4f( colorid,
                          borderColor[0],
                          borderColor[1],
                          borderColor[2],
                          borderColor[3] ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, border_line_vbo ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 24, border, GL_STATIC_DRAW ) );

    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glLineWidth( 1.0f ) ); // Apple does not support fractional line widths

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, 4 );

    GL_EXEC( glEnableVertexAttribArray( 0 ) );
    GL_EXEC( glBindVertexArray( border_line_vao ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, static_cast <GLsizei> ( 8 ) ) );
}

const ViewportPointsWithColors& ViewportGL::getPointsWithColors() const
{
    return previewPoints_;
}

const ViewportLinesWithColors& ViewportGL::getLinesWithColors() const
{
    return previewLines_;
}

void ViewportGL::setPointsWithColors( const ViewportPointsWithColors& pointsWithColors )
{
    if ( previewPoints_ == pointsWithColors )
        return;
    previewPoints_ = pointsWithColors;
    points_dirty = true;
}

void ViewportGL::setLinesWithColors( const ViewportLinesWithColors& linesWithColors )
{
    if ( previewLines_ == linesWithColors )
        return;
    previewLines_ = linesWithColors;
    lines_dirty = true;
}

void ViewportGL::fillViewport( const Vector4i& viewport, const Color& color ) const
{
    if ( !inited_ )
        return;
    // The glScissor call ensures we only clear this core's buffers,
    // (in case the user wants different background colors in each viewport.)
    GL_EXEC( glScissor( (GLsizei) viewport.x, (GLsizei) viewport.y, (GLsizei) viewport.z, (GLsizei) viewport.w ) );
    GL_EXEC( glEnable( GL_SCISSOR_TEST ) );
    auto backgroundColor = Vector4f( color );
    GL_EXEC( glClearColor( backgroundColor[0],
                           backgroundColor[1],
                           backgroundColor[2],
                           backgroundColor[3] ) );
    GL_EXEC( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
    GL_EXEC( glDisable( GL_SCISSOR_TEST ) );

}

ViewportGL::PickResults ViewportGL::pickObjects( const PickParameters& params, const std::vector<Vector2i>& picks ) const
{
    if ( !inited_ )
        return {};

    int width = params.baseRenderParams.viewport.z;
    int height = params.baseRenderParams.viewport.w;
    PickResults results( picks.size() );

    Box2i box;
    for ( const auto& pick : picks )
    {
        if ( pick.x < 0 || pick.x >= width ||
            pick.y < 0 || pick.y >= height )
            continue;
        box.include( pick );
    }

    Vector2i size;
    if ( box.valid() )
        size = box.size() + Vector2i::diagonal( 1 );

    auto resColors = pickObjectsInRect_( params, box );

    // read face and geom
    for ( int i = 0; i < results.size(); ++i )
    {
        auto pick = picks[i];
        if ( pick.x < 0 || pick.x >= width ||
            pick.y < 0 || pick.y >= height )
            continue;
        pick -= box.min;
        auto ind = pick.x + ( size.y - pick.y - 1 ) * size.x;
        const auto& color = resColors[ind].color;

        unsigned primId = color[0];
        unsigned geomId = color[1];
        float z = float(color[3])/float( 0xffffffff );

        results[i] = {geomId ,primId ,z};
    }

    // Filter results
    for ( int i = 0; i < results.size(); ++i )
    {
        if ( results[i].geomId >= params.renderVector.size() || !params.renderVector[results[i].geomId] )
            results[i] = {};
    }

    return results;
}

std::vector<unsigned> ViewportGL::findUniqueObjectsInRect( const PickParameters& params, const Box2i& rect,
                                                           int maxRenderResolutionSide ) const
{
    if ( !rect.valid() )
        return {};

    double maxBoxSide = double( maxRenderResolutionSide );
    double rWidth = double( width( rect ) );
    double rHeight = double( height( rect ) );
    Box2i updatedRect = rect;
    PickParameters updatedParams = params;
    if ( rWidth > maxBoxSide || rHeight > maxBoxSide )
    {
        double downScaleRatio = 1.0f;
        if ( rWidth > rHeight )
            downScaleRatio = maxBoxSide / rWidth;
        else
            downScaleRatio = maxBoxSide / rHeight;

        updatedRect.min = Vector2i( Vector2d( updatedRect.min ) * downScaleRatio );
        updatedRect.max = Vector2i( Vector2d( updatedRect.max ) * downScaleRatio );
        updatedParams.baseRenderParams.viewport = 
            Vector4i( Vector4d( updatedParams.baseRenderParams.viewport ) * downScaleRatio );
    }

    auto resColors = pickObjectsInRect_( updatedParams, updatedRect );

    tbb::enumerable_thread_specific<BitSet> bitSetPerThread( params.renderVector.size() );
    tbb::parallel_for( tbb::blocked_range<int>( 0, int( resColors.size() ) ),
                       [&] ( const tbb::blocked_range<int>& range )
    {
        auto& localBitSet = bitSetPerThread.local();
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto geomId = resColors[i].color[1];
            if ( geomId >= params.renderVector.size() || !params.renderVector[geomId] || localBitSet.test( geomId ) )
                continue;
            localBitSet.set( geomId );
        }
    } );

    BitSet mergeBitSet( params.renderVector.size() );
    for ( const auto& bs : bitSetPerThread )
        mergeBitSet |= bs;
    std::vector<unsigned> uniqueVec;
    uniqueVec.reserve( params.renderVector.size() );
    for ( auto i : mergeBitSet )
        uniqueVec.push_back( unsigned( i ) );

    return uniqueVec;
}

std::vector<ViewportGL::PickColor> ViewportGL::pickObjectsInRect_( const PickParameters& params, const Box2i& rect ) const
{
    Vector2i size;
    std::vector<PickColor> resColors;
    if ( rect.valid() )
    {
        size = rect.size() + Vector2i::diagonal( 1 );
        resColors.resize( size.x * size.y );
    }

    int width = params.baseRenderParams.viewport.z;
    int height = params.baseRenderParams.viewport.w;

    // https://learnopengl.com/Advanced-OpenGL/Anti-Aliasing
    unsigned int framebuffer;
    GL_EXEC( glGenFramebuffers( 1, &framebuffer ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebuffer ) );
    // create a multisampled color attachment texture
    unsigned int textureColorBufferMultiSampled;
    GL_EXEC( glGenTextures( 1, &textureColorBufferMultiSampled ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, textureColorBufferMultiSampled ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, NULL ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, 0 ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBufferMultiSampled, 0 ) );
    // create a (also multisampled) renderbuffer object for depth and stencil attachments
    unsigned int rbo;
    GL_EXEC( glGenRenderbuffers( 1, &rbo ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, rbo ) );
    GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height ) ); // GL_DEPTH_COMPONENT32 is not supported by OpenGL ES 3.0
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebuffer ) );

    // Clear the buffer
    unsigned int cClearValue[4] = { 0xffffffff,0xffffffff,0xffffffff,0xffffffff };
    GL_EXEC( glClearBufferuiv( GL_COLOR, 0, cClearValue ) );
    GL_EXEC( glClear( GL_DEPTH_BUFFER_BIT ) );
    // Save old viewport

    //draw
    GL_EXEC( glDisable( GL_BLEND ) );
    GL_EXEC( glEnable( GL_DEPTH_TEST ) );

    for ( unsigned i = 0; i < params.renderVector.size(); ++i )
    {
        auto& objPtr = params.renderVector[i];
        if ( !objPtr )
            continue;
        auto& obj = *objPtr;
        auto modelTemp = Matrix4f( obj.worldXf() );
        obj.renderForPicker( { params.baseRenderParams.viewMatrixPtr, modelTemp.data(), params.baseRenderParams.projMatrixPtr ,nullptr,
                             params.viewportId, params.clippingPlane,params.baseRenderParams.viewport }, i );
    }
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, framebuffer ) );

    if ( rect.valid() )
    {
        // read data from gpu
        GL_EXEC( glReadPixels( rect.min.x, height - rect.max.y - 1, size.x, size.y, GL_RGBA_INTEGER, GL_UNSIGNED_INT, resColors.data() ) );
    }
    // Clean up
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    GL_EXEC( glDeleteTextures( 1, &textureColorBufferMultiSampled ) );
    GL_EXEC( glDeleteFramebuffers( 1, &framebuffer ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &rbo ) );

    // Clean up
    GL_EXEC( glEnable( GL_BLEND ) );
    return resColors;
}

}
