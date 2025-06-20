#include "MRViewport.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include <MRMesh/MRVisualObject.h>
#include <MRMesh/MRLineSegm.h>
#include <MRPch/MRSpdlog.h>

namespace MR
{

bool Viewport::draw( const VisualObject& obj, DepthFunction depthFunc, RenderModelPassMask pass, bool allowAlphaSort ) const
{
    return draw( obj, obj.worldXf( id ), projM_, depthFunc, pass, allowAlphaSort );
}

bool Viewport::draw(const VisualObject& obj, const AffineXf3f& xf,
     DepthFunction depthFunc, RenderModelPassMask pass, bool allowAlphaSort ) const
{
    return draw( obj, xf, projM_, depthFunc, pass, allowAlphaSort );
}

bool Viewport::draw( const VisualObject& obj, const AffineXf3f& xf, const Matrix4f& projM,
     DepthFunction depthFunc, RenderModelPassMask pass, bool allowAlphaSort ) const
{
    Matrix4f normM;
    return obj.render( getModelRenderParams( xf, projM, &normM, depthFunc, pass, allowAlphaSort ) );
}

ModelRenderParams Viewport::getModelRenderParams( const Matrix4f & modelM, const Matrix4f & projM,
    Matrix4f * normM, DepthFunction depthFunc, RenderModelPassMask pass, bool allowAlphaSort ) const
{
    if ( normM )
    {
        auto normTemp = viewM_ * modelM;
        if ( normTemp.det() == 0 )
        {
            spdlog::warn( "Object transform is degenerate" );
            assert( false );
            *normM = normTemp;
        }
        else
            *normM = normTemp.inverse().transposed();
    }

    return ModelRenderParams
    {
        {
            getBaseRenderParams( projM ),
            modelM,
            params_.clippingPlane,
            depthFunc,
        },
        normM,
        params_.lightPosition,
        allowAlphaSort,
        pass
    };
}

void Viewport::drawLines( const std::vector<LineSegm3f>& lines, const std::vector<SegmEndColors>& colors, const LinePointImmediateRenderParams & params )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;
    // set GL_DEPTH_TEST specified for points
    GLuint lineVAO;
    GL_EXEC( glGenVertexArrays( 1, &lineVAO ) );
    GlBuffer lineBuffer, lineColorBuffer;

    // set GL_DEPTH_TEST specified for lines
    if ( params.depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( ( GLsizei )params.viewport.x, ( GLsizei )params.viewport.y,
        ( GLsizei )params.viewport.z, ( GLsizei )params.viewport.w ) );
    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( lineVAO ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::AdditionalLines );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrix.data() ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), 0.0f ) );

    GLint colorsId = GL_EXEC( glGetAttribLocation( shader, "color" ) );
    lineColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GLint positionId = GL_EXEC( glGetAttribLocation( shader, "position" ) );
    static_assert( sizeof( LineSegm3f ) == 6 * sizeof( float ), "wrong size of LineSegm3f" );
    lineBuffer.loadData( GL_ARRAY_BUFFER, lines );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, lines.size() );

    GL_EXEC( glBindVertexArray( lineVAO ) );
    GL_EXEC( glLineWidth( params.width ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, 2 * int( lines.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &lineVAO ) );
}

void Viewport::drawPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors, const LinePointImmediateRenderParams & params )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;
    // set GL_DEPTH_TEST specified for points
    GLuint pointVAO;
    // points
    GL_EXEC( glGenVertexArrays( 1, &pointVAO ) );
    GlBuffer pointBuffer, pointColorBuffer;

    if ( params.depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( ( GLsizei )params.viewport.x, ( GLsizei )params.viewport.y,
        ( GLsizei )params.viewport.z, ( GLsizei )params.viewport.w ) );
    // Send points data to GL, install points properties
    GL_EXEC( glBindVertexArray( pointVAO ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::AdditionalPoints );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrix.data() ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), 0.0f ) );

    GLint colorsId = GL_EXEC( glGetAttribLocation( shader, "color" ) );
    pointColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GLint positionId = GL_EXEC( glGetAttribLocation( shader, "position" ) );
    pointBuffer.loadData( GL_ARRAY_BUFFER, points );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointArraySize, points.size() );

    GL_EXEC( glBindVertexArray( pointVAO ) );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), params.width ) );
#else
    GL_EXEC( glPointSize( params.width ) );
#endif
    GL_EXEC( glDrawArrays( GL_POINTS, 0, int( points.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &pointVAO ) );
}

void Viewport::drawTris( const std::vector<Triangle3f>& tris, const std::vector<TriCornerColors>& colors, const ModelRenderParams& params, bool depthTest )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;
    // set GL_DEPTH_TEST specified for points
    GLuint quadVAO;
    GL_EXEC( glGenVertexArrays( 1, &quadVAO ) );
    GlBuffer quadBuffer, quadColorBuffer, quadNormalBuffer;

    // set GL_DEPTH_TEST specified for lines
    if ( depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( ( GLsizei )params.viewport.x, ( GLsizei )params.viewport.y,
        ( GLsizei )params.viewport.z, ( GLsizei )params.viewport.w ) );
    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( quadVAO ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::AdditionalQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, params.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrix.data() ) );
    auto normM = ( params.viewMatrix * params.modelMatrix ).inverse().transposed();
    if ( normM.det() == 0 )
    {
        auto norm = normM.norm();
        if ( std::isnormal( norm ) )
        {
            normM /= norm;
            normM.w = { 0, 0, 0, 1 };
        }
        else
        {
            spdlog::warn( "Object transform is degenerate" );
        }
    }
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, normM.data() ) );

    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &params.lightPos.x ) );

    GLint colorsId = GL_EXEC( glGetAttribLocation( shader, "color" ) );
    quadColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GLint normalId = GL_EXEC( glGetAttribLocation( shader, "normal" ) );
    Buffer<Vector3f> normals( tris.size() * 3 );
    for ( int i = 0; i < tris.size(); ++i )
    {
        auto* norm = &normals[i * 3];
        norm[0] = norm[1] = norm[2] = cross( tris[i][2] - tris[i][0], tris[i][1] - tris[i][0] ).normalized();
    }
    quadNormalBuffer.loadData( GL_ARRAY_BUFFER, normals );
    GL_EXEC( glVertexAttribPointer( normalId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( normalId ) );

    GLint positionId = GL_EXEC( glGetAttribLocation( shader, "position" ) );
    quadBuffer.loadData( GL_ARRAY_BUFFER, tris );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, tris.size() );

    GL_EXEC( glBindVertexArray( quadVAO ) );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 3 * int( tris.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &quadVAO ) );
}

void Viewport::drawTris( const std::vector<Triangle3f>& tris, const std::vector<Viewport::TriCornerColors>& colors, const Matrix4f& modelM, bool depthTest )
{
    Matrix4f normM;
    drawTris( tris, colors, getModelRenderParams( modelM, &normM ), depthTest );
}

} //namespace MR
