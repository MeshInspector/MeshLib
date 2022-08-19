#include "MRRenderMeshObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRPlane3.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRShadersHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMeshHolder.h"

#define RESET_VECTOR( v ) v = decltype( v ){}

namespace MR
{

RenderMeshObject::RenderMeshObject( const VisualObject& visObj )
{
    objMesh_ = dynamic_cast< const ObjectMeshHolder* >( &visObj );
    assert( objMesh_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderMeshObject::~RenderMeshObject()
{
    freeBuffers_();
}

void RenderMeshObject::render( const RenderParams& renderParams ) const
{
    if ( !objMesh_->mesh() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objMesh_->resetDirty();
        return;
    }
    update_( renderParams.viewportId );

    if ( renderParams.alphaSort )
    {
        GL_EXEC( glDepthMask( GL_FALSE ) );
        GL_EXEC( glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE ) );
        GL_EXEC( glDisable( GL_MULTISAMPLE ) );
    }
    else
    {
        GL_EXEC( glDepthMask( GL_TRUE ) );
        GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
#ifndef __EMSCRIPTEN__
        GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
    }

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

    if ( objMesh_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );
    bindMesh_( renderParams.alphaSort );

    auto shader = renderParams.alphaSort ? ShadersHolder::getShaderId( ShadersHolder::TransparentMesh ) : ShadersHolder::getShaderId( ShadersHolder::DrawMesh );
    // Send transformations to the GPU
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "onlyOddFragments" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::OnlyOddFragments, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objMesh_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "flatShading" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::FlatShading, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objMesh_->getColoringType() == ColoringType::VertsColorMap ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perFaceColoring" ), objMesh_->getColoringType() == ColoringType::FacesColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( auto fixed_colori = glGetUniformLocation( shader, "fixed_color" ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specular_exponent" ), objMesh_->getShininess() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "light_position_eye" ), 1, &renderParams.lightPos.x ) );
    GL_EXEC( glUniform4f( fixed_colori, 0.0, 0.0, 0.0, 0.0 ) );

    const auto mainColor = Vector4f( objMesh_->getFrontColor( objMesh_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "showSelectedFaces" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::SelectedFaces, renderParams.viewportId ) ) );
    const auto selectionColor = Vector4f( objMesh_->getSelectedFacesColor() );
    const auto backColor = Vector4f( objMesh_->getBackColor() );
    const auto selectionBackfacesColor = Vector4f( backColor.x * selectionColor.x, backColor.y * selectionColor.y, backColor.z * selectionColor.z, backColor.w * selectionColor.w );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionColor" ), selectionColor[0], selectionColor[1], selectionColor[2], selectionColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionBackColor" ), selectionBackfacesColor[0], selectionBackfacesColor[1], selectionBackfacesColor[2], selectionBackfacesColor[3] ) );

    // Render fill
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Faces, renderParams.viewportId ) )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

        // Texture
        GL_EXEC( auto useTexture = glGetUniformLocation( shader, "useTexture" ) );
        GL_EXEC( glUniform1i( useTexture, objMesh_->getVisualizeProperty( VisualizeMaskType::Texture, renderParams.viewportId ) ) );

        if ( renderParams.forceZBuffer )
        {
            GL_EXEC( glDepthFunc( GL_ALWAYS ) );
        }
        else
        {
            GL_EXEC( glDepthFunc( GL_LESS ) );
        }

        drawMesh_( true, renderParams.viewportId );
    }
    // Render wireframe
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Edges, renderParams.viewportId ) )
        renderMeshEdges_( renderParams );
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::BordersHighlight, renderParams.viewportId ) )
    {
        if ( dirty_ & DIRTY_BORDER_LINES )
            updateBorderLinesBuffer_();
        renderEdges_( renderParams, borderArrayObjId_, borderBufferObjId_, borderHighlightPoints_, borderPointsCount_, objMesh_->getBordersColor(), DIRTY_BORDER_LINES );
        dirty_ &= ~DIRTY_BORDER_LINES;
    }
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::SelectedEdges, renderParams.viewportId ) )
    {
        if ( dirty_ & DIRTY_EDGES_SELECTION )
            updateSelectedEdgesBuffer_();
        renderEdges_( renderParams, selectedEdgesArrayObjId_, selectedEdgesBufferObjId_, selectedEdgesPoints_, selectedPointsCount_, objMesh_->getSelectedEdgesColor(), DIRTY_EDGES_SELECTION );
        dirty_ &= ~DIRTY_EDGES_SELECTION;
    }

    if ( renderParams.alphaSort )
    {
        // enable back masks, disabled for alpha sort
        GL_EXEC( glDepthMask( GL_TRUE ) );
        GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
        GL_EXEC( glEnable( GL_MULTISAMPLE ) );
    }

    if ( memorySavingMode_ )
        resetBuffers_();
}

void RenderMeshObject::renderPicker( const BaseRenderParams& parameters, unsigned geomId ) const
{
    if ( !objMesh_->mesh() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objMesh_->resetDirty();
        return;
    }
    update_( parameters.viewportId );

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindMeshPicker_();

    auto shader = ShadersHolder::getShaderId( ShadersHolder::Picker );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrixPtr ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 3 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );

    drawMesh_( true, parameters.viewportId, true );

    if ( memorySavingMode_ )
        resetBuffers_();
}

size_t RenderMeshObject::heapBytes() const
{
    return MR::heapBytes( vertPosBufferObj_ ) 
        + MR::heapBytes( vertNormalsBufferObj_ )
        + MR::heapBytes( vertColorsBufferObj_ )
        + MR::heapBytes( vertUVBufferObj_ )
        + MR::heapBytes( facesIndicesBufferObj_ )
        + MR::heapBytes( edgesIndicesBufferObj_ )
        + MR::heapBytes( faceSelectionTexture_ )
        + MR::heapBytes( faceNormalsTexture_ )
        + MR::heapBytes( borderHighlightPoints_ )
        + MR::heapBytes( selectedEdgesPoints_ )
        + cornerNormalsCache_.heapBytes()
        + facesNormalsCache_.heapBytes();
}

const Vector<Vector3f, FaceId>& RenderMeshObject::getFacesNormals() const
{
    std::unique_lock lock( readCacheMutex_ );
    if ( dirty_ & DIRTY_FACES_NORMAL )
    {
        facesNormalsCache_ = computeFacesNormals_();
        dirty_ &= ~DIRTY_FACES_NORMAL;
    }
    return facesNormalsCache_;
}

const Vector<TriangleCornerNormals, FaceId>& RenderMeshObject::getCornerNormals() const
{
    std::unique_lock lock( readCacheMutex_ );
    if ( dirty_ & DIRTY_CORNERS_NORMAL )
    {
        cornerNormalsCache_ = computeCornerNormals_();
        dirty_ &= ~DIRTY_CORNERS_NORMAL;
    }
    return cornerNormalsCache_;
}

void RenderMeshObject::renderEdges_( const RenderParams& renderParams, GLuint vao, GLuint vbo, const std::vector<Vector3f>& data,
    GLuint count, const Color& colorChar, unsigned dirtyValue ) const
{
    if ( !count )
        return;

    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( vao ) );

    auto shader = renderParams.alphaSort ?
        ShadersHolder::getShaderId( ShadersHolder::TransparentMeshBorder ) :
        ShadersHolder::getShaderId( ShadersHolder::MeshBorder );

    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y, renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    // colors
    auto color = Vector4f( colorChar );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "uniformColor" ),
        color[0], color[1], color[2], color[3] ) );

    // positions
    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, vbo ) );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );
    if ( dirty_ & dirtyValue )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( Vector3f ) * data.size(), data.data(), GL_DYNAMIC_DRAW ) );
        dirty_ ^= dirtyValue;
    }
    GL_EXEC( glBindVertexArray( vao ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, count / 2 );

    GLfloat width = objMesh_->getEdgeWidth() * 5;
    GL_EXEC( glLineWidth( GLfloat( width ) ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, count ) );
}

void RenderMeshObject::renderMeshEdges_( const RenderParams& renderParams ) const
{
    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );

    auto shader = renderParams.alphaSort ?
        ShadersHolder::getShaderId( ShadersHolder::TransparentMeshBorder ) :
        ShadersHolder::getShaderId( ShadersHolder::MeshBorder );

    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y, renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    // colors
    auto color = Vector4f( objMesh_->getEdgesColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "uniformColor" ),
        color[0], color[1], color[2], color[3] ) );

    // positions
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, vertPosBufferObj_, 3, dirty_ & DIRTY_POSITION, memorySavingMode_ && vertsCount_ != 0 );
    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, edgesIndicesBufferObjId_ ) );
    if ( meshEdgesDirty_ )
    {
        updateMeshEdgesBuffer_();
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector2i ) * edgesIndicesBufferObj_.size(), edgesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
        meshEdgesDirty_ = false;
    }

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, meshEdgesCount_ );

    GL_EXEC( glLineWidth( GLfloat( objMesh_->getEdgeWidth() ) ) );
    GL_EXEC( glDrawElements( GL_LINES, 2 * meshEdgesCount_, GL_UNSIGNED_INT, 0 ) );
}

void RenderMeshObject::bindMesh_( bool alphaSort ) const
{
    auto shader = alphaSort ? ShadersHolder::getShaderId( ShadersHolder::TransparentMesh ) : ShadersHolder::getShaderId( ShadersHolder::DrawMesh );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, vertPosBufferObj_, 3, dirty_ & DIRTY_POSITION, memorySavingMode_ && vertsCount_ != 0 );

    bool needRefreshNormals = bool( dirty_ & DIRTY_VERTS_RENDER_NORMAL ) || bool( dirty_ & DIRTY_CORNERS_RENDER_NORMAL );
    bindVertexAttribArray( shader, "normal", vertNormalsBufferObjId_, vertNormalsBufferObj_, 3, needRefreshNormals, memorySavingMode_ && vertNormalsCount_ != 0 );
    bindVertexAttribArray( shader, "K", vertColorsBufferObjId_, vertColorsBufferObj_, 4, dirty_ & DIRTY_VERTS_COLORMAP, memorySavingMode_ && vertColorsCount_ != 0 );
    bindVertexAttribArray( shader, "texcoord", vertUVBufferObjId_, vertUVBufferObj_, 2, dirty_ & DIRTY_UV, memorySavingMode_ && vertUVCount_ != 0 );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, facesIndicesBufferObjId_ ) );
    if ( meshFacesDirty_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector3i ) * facesIndicesBufferObj_.size(), facesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
        meshFacesDirty_ = false;
    }

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, texture_ ) );
    if ( dirty_ & DIRTY_TEXTURE )
    {
        const auto& texture = objMesh_->getTexture();
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

    int maxTexSize = 0;
    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize ) );
    assert( maxTexSize > 0 );

    // Diffuse
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, faceColorsTex_ ) );
    if ( dirty_ & DIRTY_PRIMITIVE_COLORMAP )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        auto facesColorMap = objMesh_->getFacesColorMap();
        auto res = calcTextureRes( int( facesColorMap.size() ), maxTexSize );
        facesColorMap.resize( res.x * res.y );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, res.x, res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, facesColorMap.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceColors" ), 1 ) );


    // Normals
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, facesNormalsTex_ ) );
    if ( dirty_ & DIRTY_FACES_RENDER_NORMAL )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        auto res = calcTextureRes( int( faceNormalsTexture_.size() ), maxTexSize );
        faceNormalsTexture_.resize( res.x * res.y );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, res.x, res.y, 0, GL_RGBA, GL_FLOAT, faceNormalsTexture_.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceNormals" ), 2 ) );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE3 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, faceSelectionTex_ ) );
    if ( dirty_ & DIRTY_SELECTION )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        auto res = calcTextureRes( int( faceSelectionTexture_.size() ), maxTexSize );
        faceSelectionTexture_.resize( res.x * res.y );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, res.x, res.y, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, faceSelectionTexture_.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 3 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
    normalsBound_ = true;
}

void RenderMeshObject::bindMeshPicker_() const
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::Picker );
    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, vertPosBufferObj_, 3, dirty_ & DIRTY_POSITION, memorySavingMode_ && vertsCount_ != 0 );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, facesIndicesBufferObjId_ ) );
    if ( meshFacesDirty_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector3i ) * facesIndicesBufferObj_.size(), facesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
        meshFacesDirty_ = false;
    }

    dirty_ &= ~DIRTY_POSITION;
    dirty_ &= ~DIRTY_FACE;
}

void RenderMeshObject::drawMesh_( bool /*solid*/, ViewportId viewportId, bool picker ) const
{
    /* Avoid Z-buffer fighting between filled triangles & wireframe lines */
    GL_EXEC( glEnable( GL_POLYGON_OFFSET_FILL ) );
    if ( ( objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::Edges )
       // intentionally do not check selected edges and borders since they are typically thicker and include not all edges
       //  | objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::SelectedEdges )
       //  | objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::BordersHighlight ) 
        ).contains( viewportId ) )
    {
        // offset triangles further with factor depending on triangle orientation to clearly see edges on top of them
        GL_EXEC( glPolygonOffset( 1.0, 1.0 ) );
    }
    else
    {
        // offset all triangles on fixed amount to avoid halo effect in flat shading mode
        GL_EXEC( glPolygonOffset( 0.0, 1.0 ) );
    }

    if ( !picker )
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, meshFacesCount_ );

    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * meshFacesCount_, GL_UNSIGNED_INT, 0 ) );

    GL_EXEC( glDisable( GL_POLYGON_OFFSET_FILL ) );
}

void RenderMeshObject::initBuffers_()
{
    // Mesh: Vertex Array Object & Buffer objects
    GL_EXEC( glGenVertexArrays( 1, &meshArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertNormalsBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertColorsBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertUVBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &facesIndicesBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &edgesIndicesBufferObjId_ ) );
    GL_EXEC( glGenTextures( 1, &texture_ ) );

    GL_EXEC( glGenTextures( 1, &faceColorsTex_ ) );

    GL_EXEC( glGenTextures( 1, &facesNormalsTex_ ) );

    GL_EXEC( glGenTextures( 1, &faceSelectionTex_ ) );

    GL_EXEC( glGenVertexArrays( 1, &meshPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &borderArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &borderBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &selectedEdgesArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &selectedEdgesBufferObjId_ ) );

    dirty_ = DIRTY_ALL;
    normalsBound_ = false;
}

void RenderMeshObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &meshArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &meshPickerArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &borderArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &selectedEdgesArrayObjId_ ) );

    GL_EXEC( glDeleteBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertNormalsBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertColorsBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertUVBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &facesIndicesBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &edgesIndicesBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &borderBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &selectedEdgesBufferObjId_ ) );

    GL_EXEC( glDeleteTextures( 1, &texture_ ) );
    GL_EXEC( glDeleteTextures( 1, &faceColorsTex_ ) );
    GL_EXEC( glDeleteTextures( 1, &faceSelectionTex_ ) );
    GL_EXEC( glDeleteTextures( 1, &facesNormalsTex_ ) );
}

void RenderMeshObject::update_( ViewportId id ) const
{
    auto mesh = objMesh_->mesh();

    MR_TIMER;
    auto objDirty = objMesh_->getDirtyFlags();
    uint32_t dirtyNormalFlag = objMesh_->getNeededNormalsRenderDirtyValue( id );
    if ( dirtyNormalFlag & DIRTY_FACES_RENDER_NORMAL && !objMesh_->creases().any() )
        dirtyNormalFlag |= DIRTY_VERTS_RENDER_NORMAL; // vertNormalsBufferObj_ should be valid no matter what normals we use
    
    // purpose of `normalsBound_` flag:
    //     objDirty == DIRTY_FACES_RENDER_NORMAL
    // call renderPicker:
    //     dirty_ = objDirty;
    //     dirtyNormalFlag == DIRTY_FACES_RENDER_NORMAL | DIRTY_VERTS_RENDER_NORMAL;
    //     dirty_ -= ( DIRTY_RENDER_NORMALS - dirtyNormalFlag ); // dirty_ -= DIRTY_CORNERS_RENDER_NORMAL
    //     vertNormalsBufferObj_ = objVertNormals;
    //     faceNormalsTexture_ = objFaceNormals;
    //     objMesh_->resetDirty();
    //     // no bind normals because picker doesn't need it
    // call render:
    //     dirty_ = objDirty;
    //     dirtyNormalFlag == 0; // because we copied normals on `renderPicker` call
    //     dirty_ -= ( DIRTY_RENDER_NORMALS - dirtyNormalFlag ); // dirty_ -= DIRTY_RENDER_NORMALS
    //     // no coping of normals (correct)
    //     objMesh_->resetDirty();
    //     // missing bind because !(dirty_ & ( DIRTY_VERTS_RENDER_NORMAL | DIRTY_CORNERS_RENDER_NORMAL ))
    // 
    // `normalsBound_` flag saves this case
    if ( dirtyNormalFlag )
        normalsBound_ = false;

    dirty_ |= objDirty;

    if ( normalsBound_ )
        dirty_ &= ~( DIRTY_RENDER_NORMALS - dirtyNormalFlag ); // it does not affect copy, `dirtyNormalFlag` does

    const auto& edgePerFace = mesh->topology.edgePerFace();
    auto numF = mesh->topology.lastValidFace() + 1;
    auto numV = mesh->topology.lastValidVert() + 1;

    const auto& uvCoords = objMesh_->getUVCoords();
    // Vertex positions
    if ( dirty_ & DIRTY_POSITION )
    {
        MR_NAMED_TIMER( "vertbased_dirty_positions" );
        vertPosBufferObj_.resize( 3 * numF );
        BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            Vector3f v[3];
            mesh->getTriPoints( f, v[0], v[1], v[2] );
            for ( int i = 0; i < 3; ++i )
                vertPosBufferObj_[ind + i] = v[i];
        } );
        vertsCount_ = vertPosBufferObj_.size();
    }
    // Normals
    if ( dirtyNormalFlag & DIRTY_CORNERS_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_corners_normals" )
        const auto& cornerNormals = objMesh_->getCornerNormals();
        vertNormalsBufferObj_.resize( 3 * numF );
        BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            const auto& cornerN = cornerNormals[f];
            for ( int i = 0; i < 3; ++i )
                vertNormalsBufferObj_[ind + i] = cornerN[i];
        } );
        vertNormalsCount_ = vertNormalsBufferObj_.size();
    }
    if ( dirtyNormalFlag & DIRTY_VERTS_RENDER_NORMAL ) // vertNormalsBufferObj_ should be valid no matter what normals we use
    {
        MR_NAMED_TIMER( "dirty_vertices_normals" )
        const auto& vertsNormals = objMesh_->getVertsNormals();
        vertNormalsBufferObj_.resize( 3 * numF );
        BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            VertId v[3];
            mesh->topology.getTriVerts( f, v );
            for ( int i = 0; i < 3; ++i )
            {
                const auto& norm = vertsNormals[v[i]];
                vertNormalsBufferObj_[ind + i] = norm;
            }
        } );
        vertNormalsCount_ = vertNormalsBufferObj_.size();
    }
    if ( dirtyNormalFlag & DIRTY_FACES_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_faces_normals" )
        const auto& faceNormals = objMesh_->getFacesNormals();
        faceNormalsTexture_.resize( faceNormals.size() );
        BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            const auto& norm = faceNormals[f];
            faceNormalsTexture_[f] = Vector4f{ norm.x,norm.y,norm.z,1.0f };
        } );
    }

    ColoringType coloringType = objMesh_->getColoringType();
    // Per-vertex material settings
    if ( dirty_ & DIRTY_VERTS_COLORMAP && coloringType == ColoringType::VertsColorMap )
    {
        const auto& vertsColorMap = objMesh_->getVertsColorMap();
        MR_NAMED_TIMER( "vert_colormap" );
        vertColorsBufferObj_.resize( 3 * numF );
        BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            VertId v[3];
            mesh->topology.getTriVerts( f, v );
            for ( int i = 0; i < 3; ++i )
                vertColorsBufferObj_[ind + i] = vertsColorMap[v[i]];
        } );
        vertColorsCount_ = vertColorsBufferObj_.size();
    }
    // Face indices
    if ( dirty_ & DIRTY_FACE )
    {
        meshFacesDirty_ = true;
        meshEdgesDirty_ = true;
        meshFacesCount_ = numF;
        meshEdgesCount_ = numF * 3;

        facesIndicesBufferObj_.resize( meshFacesCount_ );
        BitSetParallelForAll( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            if ( f >= numF )
                return;
            if ( !edgePerFace[f].valid() )
                facesIndicesBufferObj_[f] = Vector3i();
            else
                facesIndicesBufferObj_[f] = Vector3i{ ind,ind + 1,ind + 2 };
        } );
        // NOTE: mesh edges' buffer updating is delayed until it's used
    }
    // Texture coordinates
    if ( objMesh_->getVisualizeProperty( VisualizeMaskType::Texture, ViewportMask::any() ) )
    {
        assert( uvCoords.size() >= numV );
    }
    if ( dirty_ & DIRTY_UV )
    {
        if ( uvCoords.size() >= numV )
        {
            vertUVBufferObj_.resize( 3 * numF );
            BitSetParallelFor( mesh->topology.getValidFaces(), [&] ( FaceId f )
            {
                auto ind = 3 * f;
                VertId v[3];
                mesh->topology.getTriVerts( f, v );
                for ( int i = 0; i < 3; ++i )
                    vertUVBufferObj_[ind + i] = uvCoords[v[i]];
            } );
            vertUVCount_ = vertUVBufferObj_.size();
        }
        else
        {
            vertUVCount_ = 0;
        }
    }

    if ( dirty_ & DIRTY_SELECTION )
    {
        faceSelectionTexture_.resize( numF / 32 + 1 );
        const auto& selection = objMesh_->getSelectedFaces().m_bits;
        const unsigned* selectionData = ( unsigned* )selection.data();
        tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )faceSelectionTexture_.size() ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int r = range.begin(); r < range.end(); ++r )
            {
                auto& block = faceSelectionTexture_[r];
                if ( r / 2 >= selection.size() )
                {
                    block = 0;
                    continue;
                }
                block = selectionData[r];
            }
        } );
    }

    // dirty_ instead of meshgl_.dirty because:
    // meshgl_.dirty is cleared only after render of borders, and it will not render if border highlight is turned off
    if ( objDirty & DIRTY_BORDER_LINES )
    {
        // NOTE: border lines' buffer updating is delayed until it's used
    }

    if ( objDirty & DIRTY_EDGES_SELECTION )
    {
        // NOTE: selected edges' buffer updating is delayed until it's used
    }

    objMesh_->resetDirtyExeptMask( DIRTY_RENDER_NORMALS - dirtyNormalFlag );
}

void RenderMeshObject::updateMeshEdgesBuffer_() const
{
    auto mesh = objMesh_->mesh();
    const auto& edgePerFace = mesh->topology.edgePerFace();

    edgesIndicesBufferObj_.resize( meshEdgesCount_ );
    BitSetParallelForAll( mesh->topology.getValidFaces(), [&] ( FaceId f )
    {
        auto ind = 3 * f;
        if ( f >= meshFacesCount_ )
            return;
        if ( !edgePerFace[f].valid() )
        {
            for ( int i = 0; i < 3; ++i )
                edgesIndicesBufferObj_[ind + i] = Vector2i();
        }
        else
        {
            for ( int i = 0; i < 3; ++i )
                edgesIndicesBufferObj_[ind + i] = Vector2i{ ind + i, ind + ( ( i + 1 ) % 3 ) };
        }
    } );
}

void RenderMeshObject::updateBorderLinesBuffer_() const
{
    auto mesh = objMesh_->mesh();
    auto boundary = mesh->topology.findBoundary();

    borderHighlightPoints_.clear();
    for ( auto& b : boundary )
    {
        for ( auto& e : b )
        {
            borderHighlightPoints_.push_back( mesh->points[mesh->topology.org( e )] );
            borderHighlightPoints_.push_back( mesh->points[mesh->topology.dest( e )] );
        }
    }
    borderPointsCount_ = GLsizei( borderHighlightPoints_.size() );
}

void RenderMeshObject::updateSelectedEdgesBuffer_() const
{
    auto mesh = objMesh_->mesh();

    selectedEdgesPoints_.clear();
    for ( auto e : objMesh_->getSelectedEdges() )
    {
        if ( mesh->topology.hasEdge( e ) )
        {
            selectedEdgesPoints_.push_back( mesh->orgPnt( e ) );
            selectedEdgesPoints_.push_back( mesh->destPnt( e ) );
        }
    }
    selectedPointsCount_ = GLsizei( selectedEdgesPoints_.size() );
}

void RenderMeshObject::resetBuffers_() const
{
    RESET_VECTOR( vertPosBufferObj_ );
    if ( normalsBound_ )
        RESET_VECTOR( vertNormalsBufferObj_ );
    RESET_VECTOR( vertColorsBufferObj_ );
    RESET_VECTOR( vertUVBufferObj_ );
    RESET_VECTOR( facesIndicesBufferObj_ );
    RESET_VECTOR( edgesIndicesBufferObj_ );
    RESET_VECTOR( faceSelectionTexture_ );
    RESET_VECTOR( faceNormalsTexture_ );
    RESET_VECTOR( borderHighlightPoints_ );
    RESET_VECTOR( selectedEdgesPoints_ );
}

Vector<Vector3f, FaceId> RenderMeshObject::computeFacesNormals_() const
{
    if ( !objMesh_ || !objMesh_->getMesh() )
        return {};
    return computePerFaceNormals( *( objMesh_->getMesh() ) );
}

Vector<TriangleCornerNormals, FaceId> RenderMeshObject::computeCornerNormals_() const
{
    if ( !objMesh_ || !objMesh_->getMesh() )
        return {};

    const auto& creases = objMesh_->getCreases();
    return computePerCornerNormals( *( objMesh_->getMesh() ), creases.any() ? &creases : nullptr );
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectMeshHolder, RenderMeshObject )

}
