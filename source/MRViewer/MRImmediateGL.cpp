#include "MRImmediateGL.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRShadersHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRMatrix4.h"

namespace MR
{

namespace ImmediateGL
{

void drawPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors, const ImmediateGL::RenderParams& params )
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

    auto shader = ShadersHolder::getShaderId( ShadersHolder::AdditionalPoints );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrixPtr ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), 0.0f ) );

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );
    pointColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
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

void drawLines( const std::vector<LineSegm3f>& lines, const std::vector<SegmEndColors>& colors, const ImmediateGL::RenderParams& params )
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

    auto shader = ShadersHolder::getShaderId( ShadersHolder::AdditionalLines );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrixPtr ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "offset" ), 0.0f ) );

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );
    lineColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    lineBuffer.loadData( GL_ARRAY_BUFFER, lines );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, lines.size() );

    GL_EXEC( glBindVertexArray( lineVAO ) );
    GL_EXEC( glLineWidth( params.width ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, 2 * int( lines.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &lineVAO ) );
}

void drawTris( const std::vector<Tri>& tris, const std::vector<TriCornerColors>& colors, const ImmediateGL::TriRenderParams& params )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;
    // set GL_DEPTH_TEST specified for points 
    GLuint quadVAO;
    GL_EXEC( glGenVertexArrays( 1, &quadVAO ) );
    GlBuffer quadBuffer, quadColorBuffer, quadNormalBuffer;

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
    GL_EXEC( glBindVertexArray( quadVAO ) );

    auto shader = ShadersHolder::getShaderId( ShadersHolder::AdditionalQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrixPtr ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "light_position_eye" ), 1, &params.lightPos.x ) );

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );
    quadColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GL_EXEC( GLint normalId = glGetAttribLocation( shader, "normal" ) );
    Buffer<Vector3f> normals( tris.size() * 3 );
    for ( int i = 0; i < tris.size(); ++i )
    {
        auto* norm = &normals[i * 3];
        norm[0] = norm[1] = norm[2] = cross( tris[i].c - tris[i].a, tris[i].b - tris[i].a ).normalized();
    }
    quadNormalBuffer.loadData( GL_ARRAY_BUFFER, normals );
    GL_EXEC( glVertexAttribPointer( normalId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( normalId ) );

    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    quadBuffer.loadData( GL_ARRAY_BUFFER, tris );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, tris.size() );

    GL_EXEC( glBindVertexArray( quadVAO ) );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 3 * int( tris.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &quadVAO ) );
}

} //namespace ImmediateGL

} //namespace MR
