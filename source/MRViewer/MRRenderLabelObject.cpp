#include "MRRenderLabelObject.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRSymbolMesh/MRSymbolMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMatrix4.h"
#include "MRGLMacro.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRGLStaticHolder.h"
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

bool RenderLabelObject::render( const ModelRenderParams& renderParams )
{
    RenderModelPassMask desiredPass =
        !objLabel_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) ? RenderModelPassMask::NoDepthTest :
        ( objLabel_->getGlobalAlpha( renderParams.viewportId ) < 255 || objLabel_->getFrontColor( objLabel_->isSelected(), renderParams.viewportId ).a < 255 ) ? RenderModelPassMask::Transparent :
        RenderModelPassMask::Opaque;
    if ( !bool( renderParams.passMask & desiredPass ) )
        return false; // Nothing to draw in this pass.

    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLabel_->resetDirty();
        return false;
    }

    update_();

    if ( objLabel_->globalClippedByPlane( renderParams.viewportId ) )
    {
        Vector3f pos = renderParams.modelMatrix( objLabel_->getLabel().position );
        if ( dot( pos, renderParams.clipPlane.n ) > renderParams.clipPlane.d )
            return false;
    }

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

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Labels );

    // Send transformations to the GPU
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );

    auto height = objLabel_->getFontHeight();

    Vector2f modifier;
    modifier.y = height / ( SymbolMeshParams::MaxGeneratedFontHeight * renderParams.viewport.w );
    modifier.x = modifier.y * renderParams.viewport.w / renderParams.viewport.z;

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "modifier" ), modifier.x, modifier.y ) );

    const auto& pos = objLabel_->getLabel().position;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "basePos" ), pos.x, pos.y, pos.z ) );

	Vector2f shift = objLabel_->getPivotShift();
	if ( objLabel_->getVisualizeProperty( LabelVisualizePropertyType::Contour, renderParams.viewportId ) )
	{
		const auto color = Vector4f( objLabel_->getContourColor( renderParams.viewportId ) );
		GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), color[0], color[1], color[2], color[3] ) );

		auto contourFn = [&] ( Vector2f contourShift )
		{
			const Vector2f newShift = shift + contourShift;
			GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), newShift.x, newShift.y ) );
			getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, faceIndicesSize_ );
            GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
			GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( faceIndicesSize_ ), GL_UNSIGNED_INT, 0 ) );
            GL_EXEC( glDepthFunc( getDepthFunctionLEqual( DepthFunction::Default ) ) );
		};
		contourFn( Vector2f( 1, 1 ) / 2.f );
		contourFn( Vector2f( 0, 1 ) / 2.f );
		contourFn( Vector2f( -1, 1 ) / 2.f );
		contourFn( Vector2f( -1, 0 ) / 2.f );
		contourFn( Vector2f( -1, -1 ) / 2.f );
		contourFn( Vector2f( 0, -1 ) / 2.f );
		contourFn( Vector2f( 1, -1 ) / 2.f );
		contourFn( Vector2f( 1, 0 ) / 2.f );
	}
	GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shift.x, shift.y ) );

	const auto mainColor = Vector4f( objLabel_->getFrontColor( objLabel_->isSelected() ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objLabel_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, faceIndicesSize_ );

    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( faceIndicesSize_ ), GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( DepthFunction::Default ) ) );

    GL_EXEC( glDepthFunc( GL_LESS ) );

    return true;
}

void RenderLabelObject::renderSourcePoint_( const ModelRenderParams& renderParams )
{
    GL_EXEC( glBindVertexArray( srcArrayObjId_ ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Points );
    GL_EXEC( glUseProgram( shader ) );

    const std::array<Vector3f, 1> point { objLabel_->getLabel().position };
    bindVertexAttribArray( shader, "position", srcVertPosBuffer_, point, 3, dirtySrc_ );

    constexpr std::array<VertId, 1> pointIndices{ VertId( 0 ) };
    srcIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, dirtySrc_, pointIndices );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );

    const auto& mainColor = Vector4f( objLabel_->getSourcePointColor( renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objLabel_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    unsigned selTexture = 0;
    srcIndicesSelectionTex_.loadDataOpt( dirtySrc_,
        { .resolution = {1, 1, 1}, .internalFormat = GL_R32UI, .format = GL_RED_INTEGER, .type= GL_UNSIGNED_INT },
        (const char*)&selTexture );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 0 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, pointIndices.size() );

#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objLabel_->getSourcePointSize() ) );
#else
    GL_EXEC( glPointSize( objLabel_->getSourcePointSize() ) );
#endif
    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )pointIndices.size(), GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( DepthFunction::Default ) ) );

    dirtySrc_ = false;
}

void RenderLabelObject::renderBackground_( const ModelRenderParams& renderParams )
{
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Labels );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );

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
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objLabel_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    auto box = meshBox_;
    applyPadding( box, objLabel_->getBackgroundPadding() * ( box.max.y - box.min.y ) / height );
    const std::array<Vector3f, 4> corners {
        Vector3f{ box.min.x, box.min.y, 0.f },
        Vector3f{ box.max.x, box.min.y, 0.f },
        Vector3f{ box.min.x, box.max.y, 0.f },
        Vector3f{ box.max.x, box.max.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", bgVertPosBuffer_, corners, 3, dirtyBg_ );

    constexpr std::array<Vector3i, 2> bgFacesIndicesBufferObj = {
        Vector3i{ 0, 1, 2 },
        Vector3i{ 1, 2, 3 },
    };

    bgFacesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, dirtyBg_, bgFacesIndicesBufferObj );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, bgFacesIndicesBufferObj.size() );

    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_TRIANGLES, 3 * int( bgFacesIndicesBufferObj.size() ), GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( DepthFunction::Default ) ) );

    dirtyBg_ = false;
}

void RenderLabelObject::renderLeaderLine_( const ModelRenderParams& renderParams )
{
    GL_EXEC( glBindVertexArray( llineArrayObjId_ ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Labels );
    GL_EXEC( glUseProgram( shader ) );

    const auto shift = objLabel_->getPivotShift();
    auto box = meshBox_;
    applyPadding( box, objLabel_->getBackgroundPadding() * ( box.max.y - box.min.y ) / objLabel_->getFontHeight() );
    const std::array<Vector3f, 5> leaderLineVertices {
        Vector3f{ shift.x, shift.y, 0.f },
        Vector3f{ box.min.x, box.min.y, 0.f },
        Vector3f{ box.max.x, box.min.y, 0.f },
        Vector3f{ box.min.x, box.max.y, 0.f },
        Vector3f{ box.max.x, box.max.y, 0.f },
    };
    bindVertexAttribArray( shader, "position", llineVertPosBuffer_, leaderLineVertices, 3, dirtyLLine_ );

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

    llineEdgesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, dirtyLLine_, llineEdgesIndices.data(), llineEdgesIndicesSize );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );

    auto height = objLabel_->getFontHeight();

    Vector2f modifier;
    modifier.y = height / ( SymbolMeshParams::MaxGeneratedFontHeight * renderParams.viewport.w );
    modifier.x = modifier.y * renderParams.viewport.w / renderParams.viewport.z;

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "modifier" ), modifier.x, modifier.y ) );

    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shift.x, shift.y ) );

    const auto& pos = objLabel_->getLabel().position;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "basePos" ), pos.x, pos.y, pos.z ) );

    const auto mainColor = Vector4f( objLabel_->getLeaderLineColor( renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objLabel_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );


    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::LineElementsNum, llineEdgesIndicesSize );

    GL_EXEC( glLineWidth( objLabel_->getLeaderLineWidth() ) );

    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_LINES, 2 * int( llineEdgesIndicesSize ), GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( DepthFunction::Default ) ) );

    dirtyLLine_ = false;
}

void RenderLabelObject::renderPicker( const ModelBaseRenderParams&, unsigned )
{
    // no picker for labels
}

size_t RenderLabelObject::heapBytes() const
{
    return 0;
}

size_t RenderLabelObject::glBytes() const
{
    return vertPosBuffer_.size()
        + facesIndicesBuffer_.size()
        + srcVertPosBuffer_.size()
        + srcIndicesBuffer_.size()
        + srcIndicesSelectionTex_.size()
        + bgVertPosBuffer_.size()
        + bgFacesIndicesBuffer_.size()
        + llineVertPosBuffer_.size()
        + llineEdgesIndicesBuffer_.size();
}

void RenderLabelObject::forceBindAll()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;
    update_();
    bindLabel_();
}

void RenderLabelObject::bindLabel_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Labels );
    GL_EXEC( glBindVertexArray( labelArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    if ( auto mesh = objLabel_->labelRepresentingMesh() )
        bindVertexAttribArray( shader, "position", vertPosBuffer_, mesh->points.vec_, 3, dirty_ & DIRTY_POSITION );
    else
        bindVertexAttribArray( shader, "position", vertPosBuffer_, std::vector<Vector3f>{}, 3, false, vertPosBuffer_.valid() );

    auto faceIndices = loadFaceIndicesBuffer_();
    facesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, faceIndices.dirty(), faceIndices );
    dirty_ &= ~DIRTY_MESH;
}

void RenderLabelObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &labelArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( labelArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( srcArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( bgArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &llineArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( llineArrayObjId_ ) );

    dirty_ = DIRTY_ALL;
    dirtySrc_ = true;
    dirtyBg_ = true;
    dirtyLLine_ = true;
}

void RenderLabelObject::freeBuffers_()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;

    GL_EXEC( glDeleteVertexArrays( 1, &labelArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &srcArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &bgArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &llineArrayObjId_ ) );
}

void RenderLabelObject::update_()
{
    auto objDirty = objLabel_->getDirtyFlags();
    dirty_ |= objDirty;

    if ( dirty_ & DIRTY_FACE )
    {
        dirtyBg_ = true;
        dirtyLLine_ = true;
        if ( auto mesh = objLabel_->labelRepresentingMesh() )
            meshBox_ = mesh->getBoundingBox();
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

RenderBufferRef<Vector3i> RenderLabelObject::loadFaceIndicesBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACE ) || !objLabel_->labelRepresentingMesh() )
        return glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_, !facesIndicesBuffer_.valid() );

    MR_TIMER;

    const auto& mesh = objLabel_->labelRepresentingMesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_ = numF );

    BitSetParallelForAll( topology.getValidFaces(), [&] ( FaceId f )
    {
        if ( f >= numF )
            return;
        topology.getTriVerts( f, ( VertId( & )[3] ) buffer[int( f )] );
    } );

    return buffer;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLabel, RenderLabelObject )

}
