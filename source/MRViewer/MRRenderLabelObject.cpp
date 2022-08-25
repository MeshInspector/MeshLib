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

    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::LeaderLine, renderParams.viewportId ) )
        renderLeaderLine_( renderParams );
    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::SourcePoint, renderParams.viewportId ) )
        renderSourcePoint_( renderParams );
    if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::Background, renderParams.viewportId ) )
        renderBackground_( renderParams );

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
    bindVertexAttribArray( shader, "position", srcVertPosBuffer_, point.data(), point.size(), 3, dirtySrc_ );

    constexpr std::array<VertId, 1> pointIndices{ VertId( 0 ) };
    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, srcIndicesBufferObjId_ ) );
    if ( dirtySrc_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( VertId ) * pointIndices.size(), pointIndices.data(), GL_DYNAMIC_DRAW ) );
    }

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    const auto& mainColor = Vector4f( objLabel_->getSourcePointColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, srcIndicesSelectionTexId_ ) );
    if ( dirtySrc_ )
    {
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
        GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
        GL_EXEC( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
        unsigned selTexture = 0;
        GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, 1, 1, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &selTexture ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 0 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, pointIndices.size() );

#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objLabel_->getSourcePointSize() ) );
#else
    GL_EXEC( glPointSize( objLabel_->getSourcePointSize() ) );
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
    applyPadding( box, objLabel_->getBackgroundPadding() * ( box.max.y - box.min.y ) / height );
    const std::vector<Vector3f> corners {
        { box.min.x, box.min.y, 0.f },
        { box.max.x, box.min.y, 0.f },
        { box.min.x, box.max.y, 0.f },
        { box.max.x, box.max.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", bgVertPosBuffer_, corners.data(), corners.size(), 3, dirtyBg_ );

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
    applyPadding( box, objLabel_->getBackgroundPadding() * ( box.max.y - box.min.y ) / objLabel_->getFontHeight() );
    const std::vector<Vector3f> leaderLineVertices {
        { shift.x, shift.y, 0.f },
        { box.min.x, box.min.y, 0.f },
        { box.max.x, box.min.y, 0.f },
        { box.min.x, box.max.y, 0.f },
        { box.max.x, box.max.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", llineVertPosBuffer_, leaderLineVertices.data(), leaderLineVertices.size(), 3, dirtyLLine_ );

    std::array<Vector2i, 3> llineEdgesIndices {
        Vector2i{ 1, 2 },
        Vector2i{ 0, 1 },
        Vector2i{ 1, 3 },
    };
    size_t llineEdgesIndicesSize;
    const auto middleX = ( box.max.x - box.min.x ) / 2.f;
    if ( shift.x < box.min.x || box.max.x < shift.x || shift.y < box.min.y )
    {
        llineEdgesIndicesSize = 2;
        // lead to closest lower corner
        if ( shift.x < middleX )
            llineEdgesIndices[1] = Vector2i{ 0, 1 };
        else
            llineEdgesIndices[1] = Vector2i{ 0, 2 };
    }
    else if ( box.max.y < shift.y )
    {
        llineEdgesIndicesSize = 3;
        // lead to closest upper corner and then to bottom
        if ( shift.x < middleX )
        {
            llineEdgesIndices[1] = Vector2i{ 0, 3 };
            llineEdgesIndices[2] = Vector2i{ 1, 3 };
        }
        else
        {
            llineEdgesIndices[1] = Vector2i{ 0, 4 };
            llineEdgesIndices[2] = Vector2i{ 2, 4 };
        }
    }
    else
    {
        // source point is hidden
        llineEdgesIndicesSize = 1;
    }
    assert( llineEdgesIndicesSize <= llineEdgesIndices.size() );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, llineEdgesIndicesBufferObjId_ ) );
    if ( dirtyLLine_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector2i ) * llineEdgesIndicesSize, llineEdgesIndices.data(), GL_DYNAMIC_DRAW ) );
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

    const auto mainColor = Vector4f( objLabel_->getLeaderLineColor() );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, llineEdgesIndicesSize );

    GL_EXEC( glLineWidth( objLabel_->getLeaderLineWidth() ) );
    GL_EXEC( glDrawElements( GL_LINES, 2 * int( llineEdgesIndicesSize ), GL_UNSIGNED_INT, 0 ) );

    dirtyLLine_ = false;
}

void RenderLabelObject::renderPicker( const BaseRenderParams&, unsigned ) const
{
    // no picker for labels
}

size_t RenderLabelObject::heapBytes() const
{
    return MR::heapBytes( facesIndicesBufferObj_ );
}

void RenderLabelObject::bindLabel_() const
{
    auto shader = ShadersHolder::getShaderId( ShadersHolder::Labels );
    GL_EXEC( glBindVertexArray( labelArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    const auto& points = objLabel_->labelRepresentingMesh()->points.vec_;
    bindVertexAttribArray( shader, "position", vertPosBuffer_, points.data(), points.size(), 3, dirty_ & DIRTY_POSITION );
    
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
    GL_EXEC( glGenBuffers( 1, &facesIndicesBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( srcArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &srcIndicesBufferObjId_ ) );
    GL_EXEC( glGenTextures( 1, &srcIndicesSelectionTexId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &bgFacesIndicesBufferObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &llineArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( llineArrayObjId_ ) );
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
    GL_EXEC( glDeleteBuffers( 1, &facesIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &srcIndicesBufferObjId_ ) );
    GL_EXEC( glDeleteTextures( 1, &srcIndicesSelectionTexId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &bgFacesIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &llineArrayObjId_ ) );
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

    const auto backgroundPadding = objLabel_->getBackgroundPadding();
    if ( backgroundPadding != backgroundPaddingState_ )
    {
        backgroundPaddingState_ = backgroundPadding;

        dirtyBg_ = true;
        dirtyLLine_ = true;
    }

    objLabel_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLabel, RenderLabelObject )

}