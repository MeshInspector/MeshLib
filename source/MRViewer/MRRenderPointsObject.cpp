#include "MRRenderPointsObject.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRPointCloud.h"
#include "MRGLMacro.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRMeshViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

RenderPointsObject::RenderPointsObject( const VisualObject& visObj )
{
    objPoints_ = dynamic_cast< const ObjectPointsHolder* >( &visObj );
    assert( objPoints_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderPointsObject::~RenderPointsObject()
{
    freeBuffers_();
}

void RenderPointsObject::render( const RenderParams& renderParams )
{
    if ( !objPoints_->pointCloud() || !Viewer::constInstance()->isGLInitialized() )
    {
        objPoints_->resetDirty();
        return;
    }
    update_();

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

    if ( objPoints_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    bindPoints_();

    // Send transformations to the GPU

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawPoints );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objPoints_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objPoints_->getColoringType() == ColoringType::VertsColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objPoints_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "hasNormals" ), int( !objPoints_->pointCloud()->normals.empty() ) ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specular_exponent" ), objPoints_->getShininess() ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "light_position_eye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objPoints_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objPoints_->getFrontColor( objPoints_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "showSelectedVertices" ), objPoints_->getVisualizeProperty( PointsVisualizePropertyType::SelectedVertices, renderParams.viewportId ) ) );
    const auto selectionColor = Vector4f( objPoints_->getSelectedVerticesColor() );
    const auto selectionBackColor = Vector4f( backColor.x * selectionColor.x, backColor.y * selectionColor.y, backColor.z * selectionColor.z, backColor.w * selectionColor.w );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionColor" ), selectionColor[0], selectionColor[1], selectionColor[2], selectionColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionBackColor" ), selectionBackColor[0], selectionBackColor[1], selectionBackColor[2], selectionBackColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, validIndicesSize_ );

#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objPoints_->getPointSize() ) );
#else
    GL_EXEC( glPointSize( objPoints_->getPointSize() ) );
#endif
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )validIndicesSize_, GL_UNSIGNED_INT, 0 ) );
}

void RenderPointsObject::renderPicker( const BaseRenderParams& parameters, unsigned geomId )
{
    if ( !objPoints_->pointCloud() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objPoints_->resetDirty();
        return;
    }
    update_();

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindPointsPicker_();

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrixPtr ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objPoints_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objPoints_->getPointSize() ) );
#else
    GL_EXEC( glPointSize( objPoints_->getPointSize() ) );
#endif
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )validIndicesSize_, GL_UNSIGNED_INT, 0 ) );
}

size_t RenderPointsObject::heapBytes() const
{
    return 0;
}

void RenderPointsObject::bindPoints_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawPoints );
    GL_EXEC( glBindVertexArray( pointsArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBuffer_, objPoints_->pointCloud()->points.vec_, 3, dirty_ & DIRTY_POSITION );
    bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, objPoints_->pointCloud()->normals.vec_, 3, dirty_ & DIRTY_RENDER_NORMALS );
    bindVertexAttribArray( shader, "K", vertColorsBuffer_, objPoints_->getVertsColorMap().vec_, 4, dirty_ & DIRTY_VERTS_COLORMAP );

    auto validIndices = loadValidIndicesBuffer_();
    validIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, validIndices.dirty(), validIndices );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, vertSelectionTex_ ) );
    auto vertSelectionTexture = loadVertSelectionTextureBuffer_();
    if ( vertSelectionTexture.dirty() )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );

        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, vertSelectionTextureSize_.x, vertSelectionTextureSize_.y, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, vertSelectionTexture.data() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 0 ) );

    dirty_ &= ~DIRTY_MESH;
}

void RenderPointsObject::bindPointsPicker_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );
    GL_EXEC( glBindVertexArray( pointsPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBuffer_, objPoints_->pointCloud()->points.vec_, 3, dirty_ & DIRTY_POSITION );

    auto validIndices = loadValidIndicesBuffer_();
    validIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, validIndices.dirty(), validIndices );
    dirty_ &= ~DIRTY_POSITION;
}

void RenderPointsObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &pointsArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( pointsArrayObjId_ ) );

    GL_EXEC( glGenTextures( 1, &vertSelectionTex_ ) );

    GL_EXEC( glGenVertexArrays( 1, &pointsPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( pointsPickerArrayObjId_ ) );

    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize_ ) );
    assert( maxTexSize_ > 0 );

    dirty_ = DIRTY_ALL;
}

void RenderPointsObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &pointsArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &pointsPickerArrayObjId_ ) );

    GL_EXEC( glDeleteTextures( 1, &vertSelectionTex_ ) );
}

void RenderPointsObject::update_()
{
    dirty_ |= objPoints_->getDirtyFlags();
    objPoints_->resetDirty();
}

RenderBufferRef<VertId> RenderPointsObject::loadValidIndicesBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) )
        return glBuffer.prepareBuffer<VertId>( validIndicesSize_, false );

    const auto& points = objPoints_->pointCloud();
    validIndicesSize_ = (int)points->points.size();
    auto buffer = glBuffer.prepareBuffer<VertId>( validIndicesSize_ );

    const auto& validPoints = points->validPoints;
    auto firstValid = validPoints.find_first();
    if ( firstValid.valid() )
    {
        BitSetParallelForAll( validPoints, [&] ( VertId v )
        {
            if ( validPoints.test( v ) )
                buffer[v] = v;
            else
                buffer[v] = firstValid;
        });
    }

    return buffer;
}

RenderBufferRef<unsigned> RenderPointsObject::loadVertSelectionTextureBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_SELECTION ) )
        return glBuffer.prepareBuffer<unsigned>( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y, false );

    const auto& points = objPoints_->pointCloud();
    const auto numV = points->validPoints.find_last() + 1;
    auto size = numV / 32 + 1;
    vertSelectionTextureSize_ = calcTextureRes( size, maxTexSize_ );
    assert( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y >= size );
    auto buffer = glBuffer.prepareBuffer<unsigned>( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y );

    const auto& selection = objPoints_->getSelectedPoints().m_bits;
    const unsigned* selectionData = (unsigned*) selection.data();
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)buffer.size() ), [&]( const tbb::blocked_range<int>& range )
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

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectPointsHolder, RenderPointsObject )

}