#include "MRRenderLinesObject.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVector2.h"
#include "MRShadersHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

RenderLinesObject::RenderLinesObject( const VisualObject& visObj )
{
    objLines_ = dynamic_cast< const ObjectLinesHolder* >( &visObj );
    assert( objLines_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderLinesObject::~RenderLinesObject()
{
    freeBuffers_();
}

float RenderLinesObject::actualLineWidth() const
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return 0.0f;

    const auto& range = GetAvailableLineWidthRange();
    return std::clamp( objLines_->getLineWidth(), range[0], range[1] );
}

void RenderLinesObject::render( const RenderParams& renderParams ) const
{
    if ( !objLines_->polyline() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLines_->resetDirty();
        return;
    }

    update_();

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

    if ( objLines_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    bindLines_();
    auto shader = ShadersHolder::getShaderId( ShadersHolder::DrawLines );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objLines_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objLines_->getColoringType() == ColoringType::VertsColorMap ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perLineColoring" ), objLines_->getColoringType() == ColoringType::LinesColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "hasNormals" ), int( !objLines_->getVertsNormals().empty() ) ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specular_exponent" ), objLines_->getShininess() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "light_position_eye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objLines_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objLines_->getFrontColor( objLines_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 2 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, linesIndicesBufferObj_.size() );
    GL_EXEC( glLineWidth( objLines_->getLineWidth() ) );
    GL_EXEC( glDrawElements( GL_LINES, ( GLsizei )linesIndicesBufferObj_.size() * 2, GL_UNSIGNED_INT, 0 ) );

    if ( objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId ) ||
        objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId ) )
        drawPoints_( renderParams );
}

void RenderLinesObject::renderPicker( const BaseRenderParams& parameters, unsigned geomId ) const
{
    if ( !objLines_->polyline() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLines_->resetDirty();
        return;
    }
    update_();

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindLinesPicker_();

    auto shader = ShadersHolder::getShaderId( ShadersHolder::Picker );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrixPtr ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 2 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );

    GL_EXEC( glLineWidth( objLines_->getLineWidth() ) );
    GL_EXEC( glDrawElements( GL_LINES, ( GLsizei )linesIndicesBufferObj_.size() * 2, GL_UNSIGNED_INT, 0 ) );

    // Fedor: should not we draw points here as well?
}

void RenderLinesObject::bindLines_() const
{
    MR_TIMER;
    auto shader = ShadersHolder::getShaderId( ShadersHolder::DrawLines );
    GL_EXEC( glBindVertexArray( linesArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, vertPosBufferObj_, 3, dirty_ & DIRTY_POSITION );
    bindVertexAttribArray( shader, "normal", vertNormalsBufferObjId_, vertNormalsBufferObj_, 3, dirty_ & DIRTY_RENDER_NORMALS );
    bindVertexAttribArray( shader, "K", vertColorsBufferObjId_, vertColorsBufferObj_, 4, dirty_ & DIRTY_VERTS_COLORMAP );
    bindVertexAttribArray( shader, "texcoord", vertUVBufferObjId_, vertUVBufferObj_, 2, dirty_ & DIRTY_UV );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, lineIndicesBufferObjId_ ) );
    if ( dirty_ & DIRTY_FACE )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector2i ) * linesIndicesBufferObj_.size(), linesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
    }

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, texture_ ) );
    if ( dirty_ & DIRTY_TEXTURE )
    {
        const auto& texture = objLines_->getTexture();
        int warp;
        switch ( texture.warp )
        {
        default:
        case MeshTexture::WarpType::Clamp:
            warp = GL_CLAMP_TO_EDGE;
            break;
        case MeshTexture::WarpType::Repeat:
            warp = GL_REPEAT;
            break;
        case MeshTexture::WarpType::Mirror:
            warp = GL_MIRRORED_REPEAT;
            break;
        }
        int filter = texture.filter == MeshTexture::FilterType::Linear ? GL_LINEAR : GL_NEAREST;
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, warp ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, warp ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, texture.resolution.x, texture.resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.pixels.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "tex" ), 0 ) );

    // Diffuse
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, lineColorsTex_ ) );
    if ( dirty_ & DIRTY_PRIMITIVE_COLORMAP )
    {
        int maxTexSize = 0;
        GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize ) );
        assert( maxTexSize > 0 );

        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        auto linesColorMap = objLines_->getLinesColorMap();
        auto res = calcTextureRes( int( linesColorMap.size() ), maxTexSize );
        linesColorMap.resize( res.x * res.y );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, res.x, res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, linesColorMap.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "lineColors" ), 1 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
}

void RenderLinesObject::bindLinesPicker_() const
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::Picker );
    GL_EXEC( glBindVertexArray( linesPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, vertPosBufferObj_, 3, dirty_ & DIRTY_POSITION );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, lineIndicesBufferObjId_ ) );
    if ( dirty_ & DIRTY_FACE )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector2i ) * linesIndicesBufferObj_.size(), linesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
    }

    dirty_ &= ~DIRTY_POSITION;
    dirty_ &= ~DIRTY_FACE;
}

void RenderLinesObject::drawPoints_( const RenderParams& renderParams ) const
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::DrawPoints );
    GL_EXEC( glUseProgram( shader ) );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, pointsSelectionTex_ ) ); // bind empty texture
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 0 ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objLines_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objLines_->getColoringType() == ColoringType::VertsColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "hasNormals" ), 0 ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specular_exponent" ), objLines_->getShininess() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "light_position_eye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objLines_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objLines_->getFrontColor( objLines_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    const bool drawPoints = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId );
    const bool smooth = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, linesIndicesBufferObj_.size() );
    // function is executing if drawPoints == true or smooth == true => pointSize != 0
    const float pointSize = std::max( drawPoints * objLines_->getPointSize(), smooth * actualLineWidth() );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), pointSize ) );
#else
    GL_EXEC( glPointSize( pointSize ) );
#endif
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )linesIndicesBufferObj_.size() * 2, GL_UNSIGNED_INT, 0 ) );
}

void RenderLinesObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &linesArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( linesArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertNormalsBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertColorsBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertUVBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &lineIndicesBufferObjId_ ) );
    GL_EXEC( glGenTextures( 1, &texture_ ) );
    
    GL_EXEC( glGenTextures( 1, &lineColorsTex_ ) );
    GL_EXEC( glGenTextures( 1, &pointsSelectionTex_ ) );

    GL_EXEC( glGenVertexArrays( 1, &linesPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( linesPickerArrayObjId_ ) );
    dirty_ = DIRTY_ALL;

}

void RenderLinesObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &linesArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &linesPickerArrayObjId_ ) );

    GL_EXEC( glDeleteBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertNormalsBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertColorsBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertUVBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &lineIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteTextures( 1, &lineColorsTex_ ) );
    GL_EXEC( glDeleteTextures( 1, &texture_ ) );
    GL_EXEC( glDeleteTextures( 1, &pointsSelectionTex_ ) );
}

void RenderLinesObject::update_() const
{
    auto polyline = objLines_->polyline();
    const auto & topology = objLines_->polyline()->topology;

    MR_TIMER;
    auto objDirty = objLines_->getDirtyFlags();
    dirty_ |= objDirty;

    auto numL = polyline->topology.lastNotLoneEdge() + 1;
    auto undirEdgesSize = numL >> 1;
    auto numV = polyline->topology.lastValidVert() + 1;

    const auto& uvCoords = objLines_->getUVCoords();
    // Vertex positions
    if ( dirty_ & DIRTY_POSITION )
    {
        MR_NAMED_TIMER( "vertbased_dirty_positions" );
        vertPosBufferObj_.resize( 2 * numL );
        tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ),
        [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int ue = range.begin(); ue < range.end(); ++ue )
            {
                auto o = topology.org( UndirectedEdgeId( ue ) );
                auto d = topology.dest( UndirectedEdgeId( ue ) );
                if ( !o || !d )
                    continue;
                vertPosBufferObj_[2 * ue] = polyline->points[o];
                vertPosBufferObj_[2 * ue + 1] = polyline->points[d];
            }
        } );
    }
    // Normals
    const auto& vertsNormals = objLines_->getVertsNormals();
    if ( dirty_ & DIRTY_RENDER_NORMALS && vertsNormals.size() >= numV )
    {
        MR_NAMED_TIMER( "dirty_vertices_normals" )
        vertNormalsBufferObj_.resize( 2 * numL );
        tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ),
        [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int ue = range.begin(); ue < range.end(); ++ue )
            {
                auto o = topology.org( UndirectedEdgeId( ue ) );
                auto d = topology.dest( UndirectedEdgeId( ue ) );
                if ( !o || !d )
                    continue;
                vertNormalsBufferObj_[2 * ue] = vertsNormals[o];
                vertNormalsBufferObj_[2 * ue + 1] = vertsNormals[d];
            }
        } );
    }

    ColoringType coloringType = objLines_->getColoringType();
    // Per-vertex material settings
    if ( dirty_ & DIRTY_VERTS_COLORMAP && coloringType == ColoringType::VertsColorMap )
    {
        const auto& vertsColorMap = objLines_->getVertsColorMap();
        MR_NAMED_TIMER( "vert_colormap" );
        vertColorsBufferObj_.resize( 2 * numL );
        tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ),
        [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int ue = range.begin(); ue < range.end(); ++ue )
            {
                auto o = topology.org( UndirectedEdgeId( ue ) );
                auto d = topology.dest( UndirectedEdgeId( ue ) );
                if ( !o || !d )
                    continue;
                vertColorsBufferObj_[2 * ue] = vertsColorMap[o];
                vertColorsBufferObj_[2 * ue + 1] = vertsColorMap[d];
            }
        } );
    }
    // Face indices
    if ( dirty_ & DIRTY_FACE )
    {
        linesIndicesBufferObj_.resize( undirEdgesSize );
        auto lastValidEdge = ( numL - 1 ) / 2;
        tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ),
        [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int ue = range.begin(); ue < range.end(); ++ue )
            {
                auto o = topology.org( UndirectedEdgeId( ue ) );
                auto d = topology.dest( UndirectedEdgeId( ue ) );
                if ( !o || !d )
                    linesIndicesBufferObj_[ue] = Vector2i( 2 * lastValidEdge, 2 * lastValidEdge + 1 );
                else
                    linesIndicesBufferObj_[ue] = Vector2i{ 2 * ue, 2 * ue + 1 };
            }
        } );
    }
    // Texture coordinates
    if ( objLines_->getVisualizeProperty( VisualizeMaskType::Texture, ViewportMask::any() ) )
    {
        assert( uvCoords.size() >= numV );
    }
    if ( dirty_ & DIRTY_UV && uvCoords.size() >= numV )
    {
        vertUVBufferObj_.resize( 2 * numL );
        tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ),
        [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int ue = range.begin(); ue < range.end(); ++ue )
            {
                auto o = topology.org( UndirectedEdgeId( ue ) );
                auto d = topology.dest( UndirectedEdgeId( ue ) );
                if ( !o || !d )
                    continue;
                vertUVBufferObj_[2 * ue] = uvCoords[o];
                vertUVBufferObj_[2 * ue + 1] = uvCoords[d];
            }
        } );
    }
    objLines_->resetDirty();
}

const Vector2f& GetAvailableLineWidthRange()
{
    static Vector2f availableWidth = Vector2f::diagonal( -1.0f );

    if ( availableWidth[0] < 0.0f )
    {
        GL_EXEC( glGetFloatv( GL_ALIASED_LINE_WIDTH_RANGE, &availableWidth[0] ) ); // it is really 1 1 for __EMSCRIPTEN__
    }
    return availableWidth;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLinesHolder, RenderLinesObject )

}