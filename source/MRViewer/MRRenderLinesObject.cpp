#include "MRRenderLinesObject.h"
#include "MRGLMacro.h"
#include "MRCreateShader.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRViewer/MRRenderDefaultObjects.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRVector2.h"

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

bool RenderLinesObject::render( const ModelRenderParams& renderParams )
{
    RenderModelPassMask desiredPass =
        !objLines_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) ? RenderModelPassMask::NoDepthTest :
        ( objLines_->getGlobalAlpha( renderParams.viewportId ) < 255 || objLines_->getFrontColor( objLines_->isSelected(), renderParams.viewportId ).a < 255 ) ? RenderModelPassMask::Transparent :
        RenderModelPassMask::Opaque;
    if ( !bool( renderParams.passMask & desiredPass ) )
        return false; // Nothing to draw in this pass.

    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLines_->resetDirty();
        return false;
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

    render_( renderParams, false );
    if ( objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId ) ||
        objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId ) )
        render_( renderParams, true );

    return true;
}

void RenderLinesObject::renderPicker( const ModelBaseRenderParams& parameters, unsigned geomId )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objLines_->resetDirty();
        return;
    }
    update_();

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );


    renderPicker_( parameters, geomId, false );
    if ( objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, parameters.viewportId ) ||
        objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, parameters.viewportId ) )
        renderPicker_( parameters, geomId, true );
}

size_t RenderLinesObject::heapBytes() const
{
    return 0;
}

size_t RenderLinesObject::glBytes() const
{
    return
        positionsTex_.size()
        + vertColorsTex_.size()
        + lineColorsTex_.size();
}

void RenderLinesObject::forceBindAll()
{
    update_();
    bindLines_( GLStaticHolder::Lines );
    bindLines_( GLStaticHolder::LinesJoint );
}

void RenderLinesObject::render_( const ModelRenderParams& renderParams, bool points )
{
    auto shaderType = points ? GLStaticHolder::LinesJoint : GLStaticHolder::Lines;
    bindLines_( shaderType );
    auto shader = GLStaticHolder::getShaderId( shaderType );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );

    if ( !points )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( renderParams.viewport.x ), float( renderParams.viewport.y ),
            float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "width" ), objLines_->getLineWidth() ) );
    }

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objLines_->getColoringType() == ColoringType::VertsColorMap ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perLineColoring" ), objLines_->getColoringType() == ColoringType::LinesColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->globalClippedByPlane( renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objLines_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    const auto& mainColor = Vector4f( objLines_->getFrontColor( objLines_->isSelected(), renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    if ( !points )
    {
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 * lineIndicesSize_ );

        GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
        GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, lineIndicesSize_ * 6 ) );
        GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
    }
    else
    {
        const bool drawPoints = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, renderParams.viewportId );
        const bool smooth = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, renderParams.viewportId );
        // function is executing if drawPoints == true or smooth == true => pointSize != 0
        const float pointSize = std::max( drawPoints * objLines_->getPointSize(), smooth * objLines_->getLineWidth() );
#ifdef __EMSCRIPTEN__
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), pointSize ) );
#else
        GL_EXEC( glPointSize( pointSize ) );
#endif
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointArraySize, 2 * lineIndicesSize_ );

        GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
        GL_EXEC( glDrawArrays( GL_POINTS, 0, lineIndicesSize_ * 2 ) );
        GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
    }
}

void RenderLinesObject::renderPicker_( const ModelBaseRenderParams& parameters, unsigned geomId, bool points )
{
    auto shaderType = points ? GLStaticHolder::LinesJointPicker : GLStaticHolder::LinesPicker;
    bindLinesPicker_( shaderType );
    auto shader = GLStaticHolder::getShaderId( shaderType );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrix.data() ) );

    if ( !points )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( parameters.viewport.x ), float( parameters.viewport.y ),
            float( parameters.viewport.z ), float( parameters.viewport.w ) ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "width" ), objLines_->getLineWidth() ) );
    }

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objLines_->globalClippedByPlane( parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y,
        parameters.clipPlane.n.z, parameters.clipPlane.d ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );

    if ( !points )
    {
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 * lineIndicesSize_ );

        GL_EXEC( glDepthFunc( getDepthFunctionLEqual( parameters.depthFunction ) ) );
        GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, lineIndicesSize_ * 6 ) );
        GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
    }
    else
    {
        const bool drawPoints = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Points, parameters.viewportId );
        const bool smooth = objLines_->getVisualizeProperty( LinesVisualizePropertyType::Smooth, parameters.viewportId );
        // function is executing if drawPoints == true or smooth == true => pointSize != 0
        const float pointSize = std::max( drawPoints * objLines_->getPointSize(), smooth * objLines_->getLineWidth() );
#ifdef __EMSCRIPTEN__
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), pointSize ) );
#else
        GL_EXEC( glPointSize( pointSize ) );
#endif
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointArraySize, 2 * lineIndicesSize_ );

        GL_EXEC( glDepthFunc( getDepthFunctionLess( parameters.depthFunction ) ) );
        GL_EXEC( glDrawArrays( GL_POINTS, 0, lineIndicesSize_ * 2 ) );
        GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
    }
}

void RenderLinesObject::bindPositions_( GLuint shaderId )
{
    // Positions
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( dirty_ & DIRTY_POSITION )
    {
        int maxTexSize = 0;
        GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize ) );
        assert( maxTexSize > 0 );
        RenderBufferRef<Vector3f> positions;
        Vector2i res;
        if ( objLines_->polyline() )
        {
            const auto& polyline = objLines_->polyline();
            const auto& topology = polyline->topology;
            auto lastValid = topology.lastNotLoneEdge();
            auto numL = lastValid.valid() ? lastValid.undirected() + 1 : 0;

            auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
            res = calcTextureRes( int( 2 * numL ), maxTexSize );
            positions = glBuffer.prepareBuffer<Vector3f>( res.x * res.y );
            lineIndicesSize_ = numL;
            // important to be last edge org for points picker,
            // real last point will overlap invalid points so picker will return correct id
            auto lastValidEdgeOrg = lastValid.valid() ? topology.org( lastValid ) : VertId();
            tbb::parallel_for( tbb::blocked_range<int>( 0, lineIndicesSize_ ), [&] ( const tbb::blocked_range<int>& range )
            {
                for ( int ue = range.begin(); ue < range.end(); ++ue )
                {
                    auto o = topology.org( UndirectedEdgeId( ue ) );
                    auto d = topology.dest( UndirectedEdgeId( ue ) );
                    if ( !o || !d )
                    {
                        positions[2 * ue] = polyline->points[lastValidEdgeOrg];
                        positions[2 * ue + 1] = polyline->points[lastValidEdgeOrg];
                    }
                    else
                    {
                        positions[2 * ue] = polyline->points[o];
                        positions[2 * ue + 1] = polyline->points[d];
                    }
                }
            } );
        }
        positionsTex_.loadData(
            { .resolution = GlTexture2::ToResolution( res ), .internalFormat = GL_RGB32UI, .format = GL_RGB_INTEGER, .type = GL_UNSIGNED_INT },
            positions );
    }
    else
        positionsTex_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "vertices" ), 0 ) );
}

void RenderLinesObject::bindLines_( GLStaticHolder::ShaderType shaderType )
{
    MR_TIMER;
    auto shader = GLStaticHolder::getShaderId( shaderType );
    GL_EXEC( glBindVertexArray( linesArrayObjId_ ) );

    GL_EXEC( glUseProgram( shader ) );

    bindPositions_( shader );

    // Vert colors
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_VERTS_COLORMAP )
    {
        int maxTexSize = 0;
        GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize ) );
        assert( maxTexSize > 0 );

        bool useColorMap = objLines_->getColoringType() == ColoringType::VertsColorMap && !objLines_->getVertsColorMap().empty();
        RenderBufferRef<Color> textVertColorMap;
        Vector2i res;
        if ( useColorMap && objLines_->polyline() )
        {
            auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
            const auto& polyline = objLines_->polyline();
            const auto& topology = polyline->topology;
            res = calcTextureRes( (int)topology.edgeSize(), maxTexSize );
            textVertColorMap = glBuffer.prepareBuffer<Color>( res.x * res.y );
            const auto& vertsColorMap = objLines_->getVertsColorMap();
            ParallelFor( 0_ue, UndirectedEdgeId( topology.undirectedEdgeSize() ), [&]( UndirectedEdgeId ue )
            {
                auto o = topology.org( ue );
                textVertColorMap[2 * ue] = (size_t)o >= vertsColorMap.size() ? vertsColorMap.back() : vertsColorMap[o];

                auto d = topology.dest( ue );
                textVertColorMap[2 * ue + 1] = (size_t)d >= vertsColorMap.size() ? vertsColorMap.back() : vertsColorMap[d];
            } );
        }
        vertColorsTex_.loadData(
            { .resolution = GlTexture2::ToResolution( res ), .internalFormat = GL_RGBA8, .format = GL_RGBA, .type = GL_UNSIGNED_BYTE },
            textVertColorMap );
    }
    else
        vertColorsTex_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "vertColors" ), 1 ) );

    // Diffuse
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
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
            { .resolution = GlTexture2::ToResolution( res ), .internalFormat = GL_RGBA8, .format = GL_RGBA, .type = GL_UNSIGNED_BYTE },
            linesColorMap );
    }
    else
        lineColorsTex_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "lineColors" ), 2 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
}

void RenderLinesObject::bindLinesPicker_( GLStaticHolder::ShaderType shaderType )
{
    auto shader = GLStaticHolder::getShaderId( shaderType );
    GL_EXEC( glBindVertexArray( linesPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    bindPositions_( shader );

    dirty_ &= ~DIRTY_POSITION;
    dirty_ &= ~DIRTY_FACE;
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

const Vector2f& GetAvailableLineWidthRange()
{
    static Vector2f availableWidth = Vector2f( 0.5f, 15.0f );
    return availableWidth;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectLinesHolder, RenderObjectCombinator<RenderDefaultUiObject, RenderLinesObject> )

}
