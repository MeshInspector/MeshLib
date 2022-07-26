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
    //
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

    const auto corners = getCorners( objLabel_->labelRepresentingMesh()->getBoundingBox() );
    std::vector<Vector3f> bbox( std::begin( corners ), std::end( corners ) );
    bindVertexAttribArray( shader, "position", bgVertPosBufferObjId_, bbox, 3, dirtyBg_ );

    GL_EXEC( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, bgFacesIndicesBufferObjId_ ) );
    if ( dirtyBg_ )
    {
        GL_EXEC( glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( Vector3i ) * bgFacesIndicesBufferObj_.size(), bgFacesIndicesBufferObj_.data(), GL_DYNAMIC_DRAW ) );
    }

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, bgFacesIndicesBufferObj_.size() );

    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( bgFacesIndicesBufferObj_.size() ), GL_UNSIGNED_INT, 0 ) );

    dirtyBg_ = false;
}

void RenderLabelObject::renderLeaderLine_( const RenderParams& renderParams ) const
{
    //
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

    GL_EXEC( glGenVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &bgVertPosBufferObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &bgFacesIndicesBufferObjId_ ) );

    dirty_ = DIRTY_ALL;
    dirtyBg_ = true;
}

void RenderLabelObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &labelArrayObjId_ ) );

    GL_EXEC( glDeleteBuffers( 1, &vertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &facesIndicesBufferObjId_ ) );

    GL_EXEC( glDeleteVertexArrays( 1, &bgArrayObjId_ ) );

    GL_EXEC( glDeleteBuffers( 1, &bgVertPosBufferObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &bgFacesIndicesBufferObjId_ ) );
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

        bgFacesIndicesBufferObj_ = {
            { 0, 1, 2 },
            { 1, 2, 3 },
        };
        dirtyBg_ = true;
    }

    objLabel_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLabel, RenderLabelObject )

}