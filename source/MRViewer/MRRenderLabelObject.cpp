#include "MRRenderLabelObject.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRMesh/MRSymbolMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRMesh.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRShadersHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace
{
    constexpr int cBackgroundPaddingPx = 8;

    void applyPadding( MR::Box3f& box, float padding )
    {
        box.min.x -= padding;
        box.min.y -= padding;
        box.min.z -= padding;
        box.max.x += padding;
        box.max.y += padding;
        box.max.z += padding;
    }
}

namespace MR
{

RenderLabelObject::RenderLabelObject( const VisualObject& visObj )
{
    objLabel_ = dynamic_cast< const ObjectLabel* >( &visObj );
    assert( objLabel_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderLabelObject::~RenderLabelObject()
{
    freeBuffers_();
}

void RenderLabelObject::render( const RenderParams& renderParams ) const
{
    if ( !objLabel_->labelRepresentingMesh() )
        return;
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLabel_->resetDirty();
        return;
    }

    update_();

    GL_EXEC( glDepthMask( GL_TRUE ) );
    GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei ) renderParams.viewport.x, ( GLsizei ) renderParams.viewport.y,
                         ( GLsizei ) renderParams.viewport.z, ( GLsizei ) renderParams.viewport.w ) );

    if ( objLabel_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    GL_EXEC( glDepthFunc( GL_LEQUAL ) );

    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::SourcePoint, renderParams.viewportId ) )
        renderSourcePoint_( renderParams );
    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::Background, renderParams.viewportId ) )
        renderBackground_( renderParams );
    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::LeaderLine, renderParams.viewportId ) )
        renderLeaderLine_( renderParams );

    bindLabel_();

    auto shader = ShadersHolder::getShaderId( ShadersHolder::Labels );

    // Send transformations to the GPU
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    auto height = objLabel_->getFontHeight(); 

    Vector2f modifier;
    modifier.y = height / ( SymbolMeshParams::MaxGeneratedFontHeight * renderParams.viewport.w );
    modifier.x = modifier.y * renderParams.viewport.w / renderParams.viewport.z;

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "modifier" ), modifier.x, modifier.y ) );

    Vector2f shift = objLabel_->getPivotShift();
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shift.x, shift.y ) );

    const auto& pos = objLabel_->getLabel().position;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "basePos" ), pos.x, pos.y, pos.z ) );

    const auto mainColor = Vector4f( objLabel_->getFrontColor( objLabel_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, facesIndicesBufferObj_.size() );

    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( facesIndicesBufferObj_.size() ), GL_UNSIGNED_INT, 0 ) );

    GL_EXEC( glDepthFunc( GL_LESS ) );
}

void RenderLabelObject::renderSourcePoint_( const RenderParams& renderParams ) const
{
    GL_EXEC( glBindVertexArray( srcArrayObjId_ ) );

    auto shader = ShadersHolder::getShaderId( ShadersHolder::DrawPoints );
    GL_EXEC( glUseProgram( shader ) );

    const std::vector<Vector3f> point { objLabel_->getLabel().position };
    bindVertexAttribArray( shader, "position", srcVertPosBufferObjId_, point, 3, dirtySrc_ );

    constexpr std::array<VertId, 1> pointIndices{ VertId( 0 ) };
    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, srcIndicesBufferObjId_ ) );
    if ( dirtySrc_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( VertId ) * pointIndices.size(), pointIndices.data(), GL_DYNAMIC_DRAW ) );
    }

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    const auto& backColor = Vector4f( objLabel_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objLabel_->getFrontColor( objLabel_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, pointIndices.size() );

    // FIXME: parametrize it
    const float pointSize = 5.f;
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), pointSize ) );
#else
    GL_EXEC( glPointSize( pointSize ) );
#endif
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )pointIndices.size(), GL_UNSIGNED_INT, 0 ) );

    dirtySrc_ = false;
}

void RenderLabelObject::renderBackground_( const RenderParams& renderParams ) const
{
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );

    auto shader = ShadersHolder::getShaderId( ShadersHolder::Labels );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    auto height = objLabel_->getFontHeight();

    Vector2f modifier;
    modifier.y = height / ( SymbolMeshParams::MaxGeneratedFontHeight * renderParams.viewport.w );
    modifier.x = modifier.y * renderParams.viewport.w / renderParams.viewport.z;

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "modifier" ), modifier.x, modifier.y ) );

    Vector2f shift = objLabel_->getPivotShift();
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shift.x, shift.y ) );

    const auto& pos = objLabel_->getLabel().position;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "basePos" ), pos.x, pos.y, pos.z ) );

    const auto mainColor = Vector4f( objLabel_->getBackColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    auto box = objLabel_->labelRepresentingMesh()->getBoundingBox();
    applyPadding( box, cBackgroundPaddingPx * ( box.max.y - box.min.y ) / height );
    const std::vector<Vector3f> corners {
        { box.min.x, box.min.y, 0.f },
        { box.max.x, box.min.y, 0.f },
        { box.min.x, box.max.y, 0.f },
        { box.max.x, box.max.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", bgVertPosBufferObjId_, corners, 3, dirtyBg_ );

    constexpr std::array<Vector3i, 2> bgFacesIndicesBufferObj = {
        Vector3i{ 0, 1, 2 },
        Vector3i{ 1, 2, 3 },
    };

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, bgFacesIndicesBufferObjId_ ) );
    if ( dirtyBg_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector3i ) * bgFacesIndicesBufferObj.size(), bgFacesIndicesBufferObj.data(), GL_DYNAMIC_DRAW ) );
    }

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, bgFacesIndicesBufferObj.size() );

    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( bgFacesIndicesBufferObj.size() ), GL_UNSIGNED_INT, 0 ) );

    dirtyBg_ = false;
}

void RenderLabelObject::renderLeaderLine_( const RenderParams& renderParams ) const
{
    GL_EXEC( glBindVertexArray( llineArrayObjId_ ) );

    auto shader = ShadersHolder::getShaderId( ShadersHolder::Labels );
    GL_EXEC( glUseProgram( shader ) );

    const auto shift = objLabel_->getPivotShift();
    auto box = objLabel_->labelRepresentingMesh()->getBoundingBox();
    applyPadding( box, cBackgroundPaddingPx * ( box.max.y - box.min.y ) / objLabel_->getFontHeight() );
    const std::vector<Vector3f> leaderLineVertices {
        { shift.x, shift.y, 0.f },
        { box.min.x, box.min.y, 0.f },
        { box.max.x, box.min.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", llineVertPosBufferObjId_, leaderLineVertices, 3, dirtyLLine_ );

    constexpr std::array<Vector2i, 2> llineEdgesIndices{
        Vector2i{ 0, 1 },
        Vector2i{ 1, 2 },
    };
    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, llineEdgesIndicesBufferObjId_ ) );
    if ( dirtyLLine_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector2i ) * llineEdgesIndices.size(), llineEdgesIndices.data(), GL_DYNAMIC_DRAW ) );
    }

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    auto height = objLabel_->getFontHeight();

    Vector2f modifier;
    modifier.y = height / ( SymbolMeshParams::MaxGeneratedFontHeight * renderParams.viewport.w );
    modifier.x = modifier.y * renderParams.viewport.w / renderParams.viewport.z;

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "modifier" ), modifier.x, modifier.y ) );

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shift.x, shift.y ) );

    const auto& pos = objLabel_->getLabel().position;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "basePos" ), pos.x, pos.y, pos.z ) );

    const auto mainColor = Vector4f( objLabel_->getFrontColor( objLabel_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, llineEdgesIndices.size() );

    // FIXME: parametrize it
    const float lineWidth = 1.f;
    GL_EXEC( glLineWidth( lineWidth ) );
    GL_EXEC( glDrawElements( GL_LINES, 2 * int( llineEdgesIndices.size() ), GL_UNSIGNED_INT, 0 ) );

    dirtyLLine_ = false;
}

void RenderLabelObject::renderPicker( const BaseRenderParams&, unsigned ) const
{
    // no picker for labels
}

void RenderLabelObject::bindLabel_() const
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::Labels );
    GL_EXEC( glBindVertexArray( labelArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    bindVertexAttribArray( shader, "position", vertPosBufferObjId_, objLabel_->labelRepresentingMesh()->points.vec_, 3, dirty_ & DIRTY_POSITION );
    
    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, facesIndicesBufferObjId_ ) );
    if ( dirty_ & DIRTY_FACE )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector3i ) * facesIndicesBufferObj_.size(), facesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
    }

    dirty_ &= ~DIRTY_MESH;
}

void RenderLabelObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &labelArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( labelArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &facesIndicesBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( srcArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &srcVertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &srcIndicesBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &bgVertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &bgFacesIndicesBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &llineArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( llineArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &llineVertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &llineEdgesIndicesBufferObjId_ ) );

    dirty_ = DIRTY_ALL;
    dirtySrc_ = true;
    dirtyBg_ = true;
    dirtyLLine_ = true;
}

void RenderLabelObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;

    GL_EXEC( glDeleteVertexArrays( 1, &labelArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &facesIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &srcVertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &srcIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &bgVertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &bgFacesIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &llineArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &llineVertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &llineEdgesIndicesBufferObjId_ ) );
}

void RenderLabelObject::update_() const
{
    auto mesh = objLabel_->labelRepresentingMesh();

    MR_TIMER;
    auto objDirty = objLabel_->getDirtyFlags();
    dirty_ |= objDirty;

    auto numF = mesh->topology.lastValidFace() + 1;
    // Face indices
    if ( dirty_ & DIRTY_FACE )
    {
        facesIndicesBufferObj_.resize( numF );
        BitSetParallelForAll( mesh->topology.getValidFaces(), [&] ( FaceId f )
        {
            if ( f >= numF )
                return;
            mesh->topology.getTriVerts( f, ( VertId( & )[3] ) facesIndicesBufferObj_[int( f )] );
        } );

        dirtyBg_ = true;
        dirtyLLine_ = true;
    }

    const auto position = objLabel_->getLabel().position;
    if ( position != positionState_ )
    {
        positionState_ = position;

        dirtySrc_ = true;
    }

    const auto pivotPoint = objLabel_->getPivotPoint();
    if ( pivotPoint != pivotPointState_ || dirty_ & DIRTY_POSITION )
    {
        pivotPointState_ = pivotPoint;

        dirtyLLine_ = true;
    }

    objLabel_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLabel, RenderLabelObject )

}