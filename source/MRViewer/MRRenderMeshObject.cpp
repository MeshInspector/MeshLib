#include "MRRenderMeshObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRShadersHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

RenderMeshObject::RenderMeshObject( const VisualObject& visObj )
{
    objMesh_ = dynamic_cast< const ObjectMeshHolder* >( &visObj );
    assert( objMesh_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
    bufferMode_ = MemoryEfficient;
}

RenderMeshObject::~RenderMeshObject()
{
    freeBuffers_();
}

void RenderMeshObject::render( const RenderParams& renderParams )
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
        renderEdges_( renderParams, borderArrayObjId_, borderBuffer_, objMesh_->getBordersColor(), DIRTY_BORDER_LINES );
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::SelectedEdges, renderParams.viewportId ) )
        renderEdges_( renderParams, selectedEdgesArrayObjId_, selectedEdgesBuffer_, objMesh_->getSelectedEdgesColor(), DIRTY_EDGES_SELECTION );

    if ( renderParams.alphaSort )
    {
        // enable back masks, disabled for alpha sort
        GL_EXEC( glDepthMask( GL_TRUE ) );
        GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
        GL_EXEC( glEnable( GL_MULTISAMPLE ) );
    }
}

void RenderMeshObject::renderPicker( const BaseRenderParams& parameters, unsigned geomId )
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
}

size_t RenderMeshObject::heapBytes() const
{
    return 0;
}

void RenderMeshObject::renderEdges_( const RenderParams& renderParams, GLuint vao, GlBuffer & vbo, const Color& colorChar, uint32_t dirtyFlag )
{
    RenderBufferRef<Vector3f> buffer;
    switch ( dirtyFlag )
    {
    case DIRTY_BORDER_LINES:
        buffer = loadBorderHighlightPointsBuffer_();
        break;
    case DIRTY_EDGES_SELECTION:
        buffer = loadSelectedEdgePointsBuffer_();
        break;
    default:
        break;
    }
    if ( !buffer.glSize() )
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
    bindVertexAttribArray( shader, "position", vbo, buffer, 3, buffer.dirty(), true );
    dirty_ &= ~dirtyFlag;

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineArraySize, buffer.glSize() / 2 );

    GLfloat width = objMesh_->getEdgeWidth() * 5;
    GL_EXEC( glLineWidth( GLfloat( width ) ) );
    GL_EXEC( glDrawArrays( GL_LINES, 0, int( buffer.glSize() ) ) );
}

void RenderMeshObject::renderMeshEdges_( const RenderParams& renderParams )
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
    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto edges = loadEdgeIndicesBuffer_();
    edgesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, edges.dirty(), edges );
    dirtyEdges_ = false;

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, edgeIndicesSize_ );

    GL_EXEC( glLineWidth( GLfloat( objMesh_->getEdgeWidth() ) ) );
    GL_EXEC( glDrawElements( GL_LINES, int( 2 * edgeIndicesSize_ ), GL_UNSIGNED_INT, nullptr ) );
}

void RenderMeshObject::bindMesh_( bool alphaSort )
{
    auto shader = alphaSort ? ShadersHolder::getShaderId( ShadersHolder::TransparentMesh ) : ShadersHolder::getShaderId( ShadersHolder::DrawMesh );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto normals = loadVertNormalsBuffer_();
    bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, normals, 3, normals.dirty(), normals.glSize() != 0 );

    auto colormaps = loadVertColorsBuffer_();
    bindVertexAttribArray( shader, "K", vertColorsBuffer_, colormaps, 4, colormaps.dirty(), colormaps.glSize() != 0 );

    auto uvs = loadVertUVBuffer_();
    bindVertexAttribArray( shader, "texcoord", vertUVBuffer_, uvs, 2, uvs.dirty(), uvs.glSize() != 0 );

    auto faces = loadFaceIndicesBuffer_();
    facesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, faces.dirty(), faces );

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
        auto res = calcTextureRes( int( facesColorMap.size() ), maxTexSize_ );
        facesColorMap.resize( res.x * res.y );
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, res.x, res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, facesColorMap.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceColors" ), 1 ) );

    // Normals
    auto faceNormals = loadFaceNormalsTextureBuffer_();
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, facesNormalsTex_ ) );
    if ( faceNormals.dirty() )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, faceNormalsTextureSize_.x, faceNormalsTextureSize_.y, 0, GL_RGBA, GL_FLOAT, faceNormals.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceNormals" ), 2 ) );

    // Selection
    auto faceSelection = loadFaceSelectionTextureBuffer_();
    GL_EXEC( glActiveTexture( GL_TEXTURE3 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, faceSelectionTex_ ) );
    if ( faceSelection.dirty() )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, faceSelectionTextureSize_.x, faceSelectionTextureSize_.y, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, faceSelection.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 3 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
    normalsBound_ = true;
}

void RenderMeshObject::bindMeshPicker_()
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::Picker );
    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto faces = loadFaceIndicesBuffer_();
    facesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, faces.dirty(), faces );

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
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, faceIndicesSize_ );

    GL_EXEC( glDrawElements( GL_TRIANGLES, int( 3 * faceIndicesSize_ ), GL_UNSIGNED_INT, nullptr ) );

    GL_EXEC( glDisable( GL_POLYGON_OFFSET_FILL ) );
}

void RenderMeshObject::initBuffers_()
{
    // Mesh: Vertex Array Object & Buffer objects
    GL_EXEC( glGenVertexArrays( 1, &meshArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );
    GL_EXEC( glGenTextures( 1, &texture_ ) );

    GL_EXEC( glGenTextures( 1, &faceColorsTex_ ) );

    GL_EXEC( glGenTextures( 1, &facesNormalsTex_ ) );

    GL_EXEC( glGenTextures( 1, &faceSelectionTex_ ) );

    GL_EXEC( glGenVertexArrays( 1, &meshPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &borderArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &selectedEdgesArrayObjId_ ) );

    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize_ ) );
    assert( maxTexSize_ > 0 );

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

    GL_EXEC( glDeleteTextures( 1, &texture_ ) );
    GL_EXEC( glDeleteTextures( 1, &faceColorsTex_ ) );
    GL_EXEC( glDeleteTextures( 1, &faceSelectionTex_ ) );
    GL_EXEC( glDeleteTextures( 1, &facesNormalsTex_ ) );
}

void RenderMeshObject::update_( ViewportId id )
{
    auto mesh = objMesh_->mesh();

    MR_TIMER;
    auto objDirty = objMesh_->getDirtyFlags();
    uint32_t dirtyNormalFlag = objMesh_->getNeededNormalsRenderDirtyValue( id );
    if ( dirtyNormalFlag & DIRTY_FACES_RENDER_NORMAL )
    {
        // vertNormalsBufferObj_ should be valid no matter what normals we use
        if ( !objMesh_->creases().any() )
            dirtyNormalFlag |= DIRTY_VERTS_RENDER_NORMAL;
        else
            dirtyNormalFlag |= DIRTY_CORNERS_RENDER_NORMAL;
    }
    
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

    if ( objMesh_->getColoringType() != ColoringType::VertsColorMap )
        dirty_ &= ~DIRTY_VERTS_COLORMAP;

    if ( dirty_ & DIRTY_FACE )
        dirtyEdges_ = true;

    objMesh_->resetDirtyExeptMask( DIRTY_RENDER_NORMALS - dirtyNormalFlag );
}

RenderBufferRef<Vector3f> RenderMeshObject::loadVertPosBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) )
        return glBuffer.prepareBuffer<Vector3f>( vertPosSize_, false );

    MR_NAMED_TIMER( "vertbased_dirty_positions" );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector3f>( vertPosSize_ = 3 * numF );

    BitSetParallelFor( topology.getValidFaces(), [&] ( FaceId f )
    {
        auto ind = 3 * f;
        Vector3f v[3];
        mesh->getTriPoints( f, v[0], v[1], v[2] );
        for ( int i = 0; i < 3; ++i )
            buffer[ind + i] = v[i];
    } );

    return buffer;
}

RenderBufferRef<Vector3f> RenderMeshObject::loadVertNormalsBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    if ( dirty_ & DIRTY_VERTS_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_vertices_normals" )

        auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = 3 * numF );

        const auto& vertNormals = objMesh_->getVertsNormals();
        BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
        {
            auto ind = 3 * f;
            VertId v[3];
            topology.getTriVerts( f, v );
            for ( int i = 0; i < 3; ++i )
            {
                const auto &norm = vertNormals[v[i]];
                buffer[ind + i] = norm;
            }
        } );

        return buffer;
    }
    else if ( dirty_ & DIRTY_CORNERS_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_corners_normals" )

        auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = 3 * numF );

        const auto& creases = objMesh_->creases();
        const auto cornerNormals = computePerCornerNormals( *mesh, creases.any() ? &creases : nullptr );
        BitSetParallelFor( topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            const auto& cornerN = cornerNormals[f];
            for ( int i = 0; i < 3; ++i )
                buffer[ind + i] = cornerN[i];
        } );

        return buffer;
    }
    else
    {
        return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_, false );
    }
}

RenderBufferRef<Color> RenderMeshObject::loadVertColorsBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_VERTS_COLORMAP ) )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_, false );

    MR_NAMED_TIMER( "vert_colormap" );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Color>( vertColorsSize_ = 3 * numF );

    const auto& vertsColorMap = objMesh_->getVertsColorMap();
    BitSetParallelFor( topology.getValidFaces(), [&] ( FaceId f )
    {
        auto ind = 3 * f;
        VertId v[3];
        topology.getTriVerts( f, v );
        for ( int i = 0; i < 3; ++i )
            buffer[ind + i] = vertsColorMap[v[i]];
    } );

    return buffer;
}

RenderBufferRef<UVCoord> RenderMeshObject::loadVertUVBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_UV ) )
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto numV = topology.lastValidVert() + 1;

    const auto& uvCoords = objMesh_->getUVCoords();
    if ( objMesh_->getVisualizeProperty( VisualizeMaskType::Texture, ViewportMask::any() ) )
    {
        assert( uvCoords.size() >= numV );
    }
    if ( uvCoords.size() >= numV )
    {
        auto buffer = glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 3 * numF );

        BitSetParallelFor( topology.getValidFaces(), [&] ( FaceId f )
        {
            auto ind = 3 * f;
            VertId v[3];
            topology.getTriVerts( f, v );
            for ( int i = 0; i < 3; ++i )
                buffer[ind + i] = uvCoords[v[i]];
        } );

        return buffer;
    }
    else
    {
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 0 );
    }
}

RenderBufferRef<Vector3i> RenderMeshObject::loadFaceIndicesBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACE ) )
        return glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_ = numF );

    const auto& edgePerFace = topology.edgePerFace();
    BitSetParallelForAll( topology.getValidFaces(), [&] ( FaceId f )
    {
        auto ind = 3 * f;
        if ( f >= numF )
            return;
        if ( !edgePerFace[f].valid() )
            buffer[f] = Vector3i();
        else
            buffer[f] = Vector3i{ ind, ind + 1, ind + 2 };
    } );

    return buffer;
}

RenderBufferRef<Vector2i> RenderMeshObject::loadEdgeIndicesBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !dirtyEdges_ )
        return glBuffer.prepareBuffer<Vector2i>( edgeIndicesSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector2i>( edgeIndicesSize_ = 3 * numF );

    const auto& edgePerFace = topology.edgePerFace();
    BitSetParallelForAll( topology.getValidFaces(), [&] ( FaceId f )
    {
        auto ind = 3 * f;
        if ( f >= numF )
            return;
        if ( !edgePerFace[f].valid() )
        {
            for ( int i = 0; i < 3; ++i )
                buffer[ind + i] = Vector2i();
        }
        else
        {
            for ( int i = 0; i < 3; ++i )
                buffer[ind + i] = Vector2i{ ind + i, ind + ( ( i + 1 ) % 3 ) };
        }
    } );

    return buffer;
}

RenderBufferRef<unsigned> RenderMeshObject::loadFaceSelectionTextureBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_SELECTION ) )
        return glBuffer.prepareBuffer<unsigned>( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    auto size = numF / 32 + 1;
    faceSelectionTextureSize_ = calcTextureRes( size, maxTexSize_ );
    assert( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y >= size );
    auto buffer = glBuffer.prepareBuffer<unsigned>( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y );

    const auto& selection = objMesh_->getSelectedFaces().m_bits;
    const unsigned* selectionData = ( unsigned* )selection.data();
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)buffer.size() ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int r = range.begin(); r < range.end(); ++r )
        {
            auto& block = buffer[r];
            if ( r / 2 >= selection.size() )
            {
                block = 0;
                continue;
            }
            block = selectionData[r];
        }
    } );

    return buffer;
}

RenderBufferRef<Vector4f> RenderMeshObject::loadFaceNormalsTextureBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACES_RENDER_NORMAL ) )
        return glBuffer.prepareBuffer<Vector4f>( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y, false );

    MR_NAMED_TIMER( "dirty_faces_normals" )

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    faceNormalsTextureSize_ = calcTextureRes( numF, maxTexSize_ );
    assert( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y >= numF );
    auto buffer = glBuffer.prepareBuffer<Vector4f>( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y );

    computePerFaceNormals4( *mesh, buffer.data(), buffer.size() );

    return buffer;
}

RenderBufferRef<Vector3f> RenderMeshObject::loadBorderHighlightPointsBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_BORDER_LINES ) )
        return glBuffer.prepareBuffer<Vector3f>( borderHighlightPointsSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto boundary = topology.findBoundary();
    borderHighlightPointsSize_ = 0;
    for ( const auto& b : boundary )
        borderHighlightPointsSize_ += 2 * (int)b.size();
    auto buffer = glBuffer.prepareBuffer<Vector3f>( borderHighlightPointsSize_ );

    size_t cur = 0;
    for ( auto& b : boundary )
    {
        for ( auto& e : b )
        {
            buffer[cur++] = mesh->points[mesh->topology.org( e )];
            buffer[cur++] = mesh->points[mesh->topology.dest( e )];
        }
    }
    assert( cur == buffer.size() );

    return buffer;
}

RenderBufferRef<Vector3f> RenderMeshObject::loadSelectedEdgePointsBuffer_()
{
    auto& glBuffer = ShadersHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_EDGES_SELECTION ) )
        return glBuffer.prepareBuffer<Vector3f>( selectedEdgePointsSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto selectedEdges = objMesh_->getSelectedEdges();
    for ( auto e : selectedEdges )
        if ( !topology.hasEdge( e ) )
            selectedEdges.reset( e );
    auto buffer = glBuffer.prepareBuffer<Vector3f>( selectedEdgePointsSize_ = 2 * (int)selectedEdges.count() );

    size_t cur = 0;
    for ( auto e : selectedEdges )
    {
        buffer[cur++] = mesh->orgPnt( e );
        buffer[cur++] = mesh->destPnt( e );
    }
    assert( cur == buffer.size() );

    return buffer;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectMeshHolder, RenderMeshObject )

}
