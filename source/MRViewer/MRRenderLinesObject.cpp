#include "MRRenderLinesObject.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVector2.h"
#include "MRGLStaticHolder.h"
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

void RenderLinesObject::render( const RenderParams& renderParams )
{
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
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawLines );

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

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "hasNormals" ), 0 ) ); // what is a normal of 3d line?

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objLines_->getShininess() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objLines_->getSpecularStrength() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), objLines_->getAmbientStrength() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objLines_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objLines_->getFrontColor( objLines_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 2 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, lineIndicesSize_ );
    GL_EXEC( glLineWidth( objLines_->getLineWidth() ) );
    GL_EXEC( glDrawElements( GL_LINES, ( GLsizei )lineIndicesSize_ * 2, GL_UNSIGNED_INT, 0 ) );

    if ( objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId ) ||
        objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId ) )
        drawPoints_( renderParams );
}

void RenderLinesObject::renderPicker( const BaseRenderParams& parameters, unsigned geomId )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLines_->resetDirty();
        return;
    }
    update_();

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindLinesPicker_();

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrixPtr ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 2 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );

    GL_EXEC( glLineWidth( objLines_->getLineWidth() ) );
    GL_EXEC( glDrawElements( GL_LINES, ( GLsizei )lineIndicesSize_ * 2, GL_UNSIGNED_INT, 0 ) );

    // Fedor: should not we draw points here as well?
}

size_t RenderLinesObject::heapBytes() const
{
    return 0;
}

size_t RenderLinesObject::glBytes() const
{
    return vertPosBuffer_.size()
        + vertUVBuffer_.size()
        + vertNormalsBuffer_.size()
        + vertColorsBuffer_.size()
        + lineIndicesBuffer_.size()
        + texture_.size()
        + pointsSelectionTex_.size()
        + lineColorsTex_.size();
}

void RenderLinesObject::forceBindAll()
{
    update_();
    bindLines_();
}

void RenderLinesObject::bindLines_()
{
    MR_TIMER;
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawLines );
    GL_EXEC( glBindVertexArray( linesArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto normals = loadVertNormalsBuffer_();
    bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, normals, 3, normals.dirty(), normals.glSize() != 0 );

    auto colors = loadVertColorsBuffer_();
    bindVertexAttribArray( shader, "K", vertColorsBuffer_, colors, 4, colors.dirty(), colors.glSize() != 0 );

    auto uvs = loadVertUVBuffer_();
    bindVertexAttribArray( shader, "texcoord", vertUVBuffer_, uvs, 2, uvs.dirty(), uvs.glSize() != 0 );

    auto lineIndices = loadLineIndicesBuffer_();
    lineIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, lineIndices.dirty(), lineIndices );

    const auto& texture = objLines_->getTexture();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    texture_.loadDataOpt( dirty_ & DIRTY_TEXTURE,
        { 
            .resolution = texture.resolution,
            .internalFormat = GL_RGBA,
            .format = GL_RGBA,
            .type = GL_UNSIGNED_BYTE,
            .wrap = texture.wrap,
            .filter = texture.filter
        },
        texture.pixels );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "tex" ), 0 ) );

    // Diffuse
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_PRIMITIVE_COLORMAP )
    {
        int maxTexSize = 0;
        GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize ) );
        assert( maxTexSize > 0 );

        // TODO: avoid copying if no need to resize, and avoid double copying if resize is needed
        auto linesColorMap = objLines_->getLinesColorMap();
        auto res = calcTextureRes( int( linesColorMap.size() ), maxTexSize );
        linesColorMap.resize( res.x * res.y );
        lineColorsTex_.loadData( 
            { .resolution = res, .internalFormat = GL_RGBA8, .format = GL_RGBA, .type= GL_UNSIGNED_BYTE },
            linesColorMap );
    }
    else
        lineColorsTex_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "lineColors" ), 1 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
}

void RenderLinesObject::bindLinesPicker_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );
    GL_EXEC( glBindVertexArray( linesPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto lineIndices = loadLineIndicesBuffer_();
    lineIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, lineIndices.dirty(), lineIndices );

    dirty_ &= ~DIRTY_POSITION;
    dirty_ &= ~DIRTY_FACE;
}

void RenderLinesObject::drawPoints_( const RenderParams& renderParams )
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawPoints );
    GL_EXEC( glUseProgram( shader ) );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    // bind empty texture
    if ( !pointsSelectionTex_.valid() )
        pointsSelectionTex_.gen();
    pointsSelectionTex_.bind();
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

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objLines_->getShininess() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objLines_->getSpecularStrength() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), objLines_->getAmbientStrength() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objLines_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objLines_->getFrontColor( objLines_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    const bool drawPoints = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId );
    const bool smooth = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, lineIndicesSize_ );
    // function is executing if drawPoints == true or smooth == true => pointSize != 0
    const float pointSize = std::max( drawPoints * objLines_->getPointSize(), smooth * actualLineWidth() );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), pointSize ) );
#else
    GL_EXEC( glPointSize( pointSize ) );
#endif
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )lineIndicesSize_ * 2, GL_UNSIGNED_INT, 0 ) );
}

void RenderLinesObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &linesArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( linesArrayObjId_ ) );
    
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
}

void RenderLinesObject::update_()
{
    dirty_ |= objLines_->getDirtyFlags();
    objLines_->resetDirty();
}

RenderBufferRef<Vector3f> RenderLinesObject::loadVertPosBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) || !objLines_->polyline() )
        return glBuffer.prepareBuffer<Vector3f>( vertPosSize_, false );

    MR_NAMED_TIMER( "vertbased_dirty_positions" );

    const auto& polyline = objLines_->polyline();
    const auto& topology = polyline->topology;
    auto numL = topology.lastNotLoneEdge() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector3f>( vertPosSize_ = 2 * numL );

    auto undirEdgesSize = numL >> 1;
    tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int ue = range.begin(); ue < range.end(); ++ue )
        {
            auto o = topology.org( UndirectedEdgeId( ue ) );
            auto d = topology.dest( UndirectedEdgeId( ue ) );
            if ( !o || !d )
                continue;
            buffer[2 * ue] = polyline->points[o];
            buffer[2 * ue + 1] = polyline->points[d];
        }
    } );

    return buffer;
}

RenderBufferRef<Vector3f> RenderLinesObject::loadVertNormalsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_RENDER_NORMALS ) )
        return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_, false );

    // always 0!
    return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = 0 );
}

RenderBufferRef<Color> RenderLinesObject::loadVertColorsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_VERTS_COLORMAP ) || !objLines_->polyline() )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_, false );

    auto coloringType = objLines_->getColoringType();
    if ( coloringType != ColoringType::VertsColorMap )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_ = 0 );

    MR_NAMED_TIMER( "vert_colormap" );

    const auto& polyline = objLines_->polyline();
    const auto& topology = polyline->topology;
    auto numL = topology.lastNotLoneEdge() + 1;
    auto buffer = glBuffer.prepareBuffer<Color>( vertColorsSize_ = 2 * numL );

    auto undirEdgesSize = numL >> 1;
    const auto& vertsColorMap = objLines_->getVertsColorMap();
    tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int ue = range.begin(); ue < range.end(); ++ue )
        {
            auto o = topology.org( UndirectedEdgeId( ue ) );
            auto d = topology.dest( UndirectedEdgeId( ue ) );
            if ( !o || !d )
                continue;
            buffer[2 * ue] = vertsColorMap[o];
            buffer[2 * ue + 1] = vertsColorMap[d];
        }
    } );

    return buffer;
}

RenderBufferRef<UVCoord> RenderLinesObject::loadVertUVBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_UV ) || !objLines_->polyline() )
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_, false );

    const auto& polyline = objLines_->polyline();
    const auto& topology = polyline->topology;
    auto numV = topology.lastValidVert() + 1;

    const auto& uvCoords = objLines_->getUVCoords();
    if ( objLines_->getVisualizeProperty( VisualizeMaskType::Texture, ViewportMask::any() ) )
    {
        assert( uvCoords.size() >= numV );
    }
    if ( uvCoords.size() < numV )
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 0 );

    auto numL = topology.lastNotLoneEdge() + 1;
    auto buffer = glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 2 * numL );

    auto undirEdgesSize = numL >> 1;
    tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int ue = range.begin(); ue < range.end(); ++ue )
        {
            auto o = topology.org( UndirectedEdgeId( ue ) );
            auto d = topology.dest( UndirectedEdgeId( ue ) );
            if ( !o || !d )
                continue;
            buffer[2 * ue] = uvCoords[o];
            buffer[2 * ue + 1] = uvCoords[d];
        }
    } );

    return buffer;
}

RenderBufferRef<Vector2i> RenderLinesObject::loadLineIndicesBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACE ) || !objLines_->polyline() )
        return glBuffer.prepareBuffer<Vector2i>( lineIndicesSize_, !lineIndicesBuffer_.valid() );

    const auto& polyline = objLines_->polyline();
    const auto& topology = polyline->topology;
    auto numL = topology.lastNotLoneEdge() + 1;
    auto undirEdgesSize = numL >> 1;
    auto buffer = glBuffer.prepareBuffer<Vector2i>( lineIndicesSize_ = undirEdgesSize );

    auto lastValidEdge = ( numL - 1 ) / 2;
    tbb::parallel_for( tbb::blocked_range<int>( 0, undirEdgesSize ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int ue = range.begin(); ue < range.end(); ++ue )
        {
            auto o = topology.org( UndirectedEdgeId( ue ) );
            auto d = topology.dest( UndirectedEdgeId( ue ) );
            if ( !o || !d )
                buffer[ue] = Vector2i( 2 * lastValidEdge, 2 * lastValidEdge + 1 );
            else
                buffer[ue] = Vector2i{ 2 * ue, 2 * ue + 1 };
        }
    } );

    return buffer;
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