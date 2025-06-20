#include "MRViewportGL.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRMatrix4.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

static_assert( sizeof( Vector3f ) == 3 * sizeof( float ), "wrong size of Vector3f" );
static_assert( sizeof( Vector4f ) == 4 * sizeof( float ), "wrong size of Vector4f" );
static_assert( sizeof( SegmEndColors ) == 8 * sizeof( float ), "wrong size of SegmEndColors" );

static Box2i roundBox( const Box2f & rectf )
{
    return
    {
        { (int)std::lround( rectf.min.x ), (int)std::lround( rectf.min.y ) },
        { (int)std::lround( rectf.max.x ), (int)std::lround( rectf.max.y ) }
    };
}

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

    GL_EXEC( glDeleteVertexArrays( 1, &add_line_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &add_line_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &add_line_colors_vbo ) );

    GL_EXEC( glDeleteVertexArrays( 1, &add_point_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &add_point_vbo ) );
    GL_EXEC( glDeleteBuffers( 1, &add_point_colors_vbo ) );

    GL_EXEC( glDeleteVertexArrays( 1, &border_line_vao ) );
    GL_EXEC( glDeleteBuffers( 1, &border_line_vbo ) );

    pickFBO_.del();
    inited_ = false;
}

void ViewportGL::drawBorder( const Box2f& rectf, const Color& color ) const
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

    const auto rect = roundBox( rectf );
    GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    GL_EXEC( glViewport( (GLsizei) rect.min.x, (GLsizei) rect.min.y,
                         (GLsizei) width( rect ), (GLsizei) height( rect ) ) );

    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( border_line_vao ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ViewportBorder );
    GL_EXEC( glUseProgram( shader ) );

    GLint colorid = GL_EXEC( glGetUniformLocation( shader, "user_color" ) );
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

void ViewportGL::fillViewport( const Box2f& rectf, const Color& color ) const
{
    if ( !inited_ )
        return;

    const auto rect = roundBox( rectf );
    // The glScissor call ensures we only clear this core's buffers,
    // (in case the user wants different background colors in each viewport.)
    GL_EXEC( glScissor( (GLsizei) rect.min.x, (GLsizei) rect.min.y,
                        (GLsizei) width( rect ), (GLsizei) height( rect ) ) );
    GL_EXEC( glEnable( GL_SCISSOR_TEST ) );
    auto backgroundColor = Vector4f( color );
    GL_EXEC( glClearColor( backgroundColor[0],
                           backgroundColor[1],
                           backgroundColor[2],
                           backgroundColor[3] ) );
    GL_EXEC( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
    GL_EXEC( glDisable( GL_SCISSOR_TEST ) );

}

bool ViewportGL::checkInit() const
{
    return inited_;
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

        results[i] = { {geomId ,primId },z };
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

ViewportGL::ScaledPickRes ViewportGL::pickObjectsInRect( const PickParameters& params, const Box2i& rect, int maxRenderResolutionSide ) const
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
    BasePickResults res( resColors.size() );
    tbb::parallel_for( tbb::blocked_range<int>( 0, int( resColors.size() ) ),
                   [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto geomId = resColors[i].color[1];
            if ( geomId >= params.renderVector.size() || !params.renderVector[geomId] )
                continue;
            res[i].geomId = geomId;
            res[i].primId = resColors[i].color[0];
        }
    } );
    return { res,updatedRect };
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

    pickFBO_.resize( { width,height } );

    pickFBO_.bind( false );

    // Clear the buffer

    if ( rect.valid() )
    {
        GL_EXEC( glScissor( rect.min.x, height - rect.max.y - 1, size.x, size.y ) );
        GL_EXEC( glEnable( GL_SCISSOR_TEST ) );
    }

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
        auto modelTemp = Matrix4f( obj.worldXf( params.baseRenderParams.viewportId ) );
        obj.renderForPicker( {
            params.baseRenderParams,
            modelTemp,
            params.clippingPlane }, i );
    }
    pickFBO_.bind( true );

    if ( rect.valid() )
    {
        // read data from gpu
        GL_EXEC( glReadPixels( rect.min.x, height - rect.max.y - 1, size.x, size.y, GL_RGBA_INTEGER, GL_UNSIGNED_INT, resColors.data() ) );
    }
    // Clean up
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // Clean up
    GL_EXEC( glEnable( GL_BLEND ) );
    if ( rect.valid() )
    {
        GL_EXEC( glDisable( GL_SCISSOR_TEST ) );
    }
    return resColors;
}

void ViewportGL::PickTextureFrameBuffer::resize( const Vector2i& size )
{
    if ( size == Vector2i() || size == size_ )
        return;
    del();
    size_ = size;
    GL_EXEC( glGenFramebuffers( 1, &framebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebuffer_ ) );
    // create a color attachment texture
    GL_EXEC( glGenTextures( 1, &colorTexture_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, colorTexture_ ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, size_.x, size_.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, NULL ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, 0 ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture_, 0 ) );
    // create a renderbuffer object for depth
    GL_EXEC( glGenRenderbuffers( 1, &renderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, renderbuffer_ ) );
    GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, size_.x, size_.y ) ); // GL_DEPTH_COMPONENT32 is not supported by OpenGL ES 3.0
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void ViewportGL::PickTextureFrameBuffer::del()
{
    if ( framebuffer_ == 0 )
        return;
    GL_EXEC( glDeleteTextures( 1, &colorTexture_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &framebuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &renderbuffer_ ) );
}

void ViewportGL::PickTextureFrameBuffer::bind( bool read )
{
    if ( framebuffer_ == 0 )
        return;
    if ( !read )
    {
        GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebuffer_ ) );
    }
    else
    {
        GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, framebuffer_ ) );
    }
}

}
