#ifndef __EMSCRIPTEN__
#include "MRRenderVolumeObject.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRGLStaticHolder.h"
#include "MRMesh/MRFloatGrid.h"

namespace MR
{

RenderVolumeObject::RenderVolumeObject( const VisualObject& visObj )
{
    objVoxels_ = dynamic_cast< const ObjectVoxels* >( &visObj );
    assert( objVoxels_ );
    if ( getViewerInstance().isGLInitialized() )
        initBuffers_();
}

RenderVolumeObject::~RenderVolumeObject()
{
    freeBuffers_();
}

void RenderVolumeObject::render( const RenderParams& renderParams )
{
    render_( renderParams, unsigned( ~0 ) );
}

void RenderVolumeObject::renderPicker( const BaseRenderParams& renderParams, unsigned geomId )
{
    Vector3f ligthPos;
    RenderParams params{ renderParams, ligthPos };
    render_( params, geomId );
}

size_t RenderVolumeObject::heapBytes() const
{
    return 0;
}

size_t RenderVolumeObject::glBytes() const
{
    return volume_.size()
        + denseMap_.size();
}

void RenderVolumeObject::forceBindAll()
{
    update_();
    bindVolume_( true );
}

void RenderVolumeObject::render_( const RenderParams& renderParams, unsigned geomId )
{
    if ( !getViewerInstance().isGLInitialized() )
    {
        objVoxels_->resetDirty();
        return;
    }
    update_();

    bool picker = geomId != unsigned( ~0 );
    // Initialize uniform
    if ( picker )
    {
        GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0,
            ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );
    }
    else
    {
        GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
            ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );
    }

    if ( objVoxels_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    bindVolume_( picker );
    auto shader = picker ? GLStaticHolder::getShaderId( GLStaticHolder::VolumePicker ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    if ( !picker )
    {
        if ( renderParams.normMatrixPtr )
        {
            GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr->data() ) );
        }
        GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &renderParams.lightPos.x ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objVoxels_->getShininess() ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objVoxels_->getSpecularStrength() ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), objVoxels_->getAmbientStrength() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ),
        objVoxels_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    if ( geomId != unsigned( ~0 ) )
    {
        GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );
    }

    if ( picker )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( 0 ), float( 0 ),
            float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    }
    else
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( renderParams.viewport.x ), float( renderParams.viewport.y ),
            float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    }

    const auto& voxelSize = objVoxels_->vdbVolume().voxelSize;
    const auto& minCorner = objVoxels_->getActiveBounds();
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "minCorner" ), float( minCorner.min.x ), float( minCorner.min.y ), float( minCorner.min.z ) ) );
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "voxelSize" ), voxelSize.x, voxelSize.y, voxelSize.z ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "step" ), std::min( { voxelSize.x, voxelSize.y, voxelSize.z } ) ) );

    constexpr std::array<float, 24> cubePoints =
    {
        0.0f,0.0f,0.0f,
        0.0f,1.0f,0.0f,
        1.0f,1.0f,0.0f,
        1.0f,0.0f,0.0f,
        0.0f,0.0f,1.0f,
        0.0f,1.0f,1.0f,
        1.0f,1.0f,1.0f,
        1.0f,0.0f,1.0f
    };
    constexpr std::array<unsigned, 36> cubeTriangles =
    {
        0,1,2,
        2,3,0,
        0,4,5,
        5,1,0,
        0,3,7,
        7,4,0,
        6,5,4,
        4,7,6,
        1,5,6,
        6,2,1,
        6,7,3,
        3,2,6
    };

    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );
    bindVertexAttribArray( shader, "position", volumeVertsBuffer_, cubePoints, 3, !volumeVertsBuffer_.valid() );
    volumeIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, !volumeIndicesBuffer_.valid(), 
        cubeTriangles.data(), cubeTriangles.size() );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 12 );
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
    GL_EXEC( glEnable( GL_CULL_FACE ) );
    GL_EXEC( glCullFace( GL_BACK ) );

    // currently only less supported for volume rendering
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFuncion::Less ) ) );
    GL_EXEC( glDrawElements( GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFuncion::Default ) ) );

    GL_EXEC( glDisable( GL_CULL_FACE ) );
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
}

void RenderVolumeObject::bindVolume_( bool picker )
{
    auto shader = picker ? GLStaticHolder::getShaderId( GLStaticHolder::VolumePicker ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    const auto& params = objVoxels_->getVolumeRenderingParams();

    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( dirty_ & DIRTY_PRIMITIVES )
    {
        auto volume = objVoxels_->getVolumeRenderingData();
        if ( !volume )
        {
            objVoxels_->prepareDataForVolumeRendering();
            volume = objVoxels_->getVolumeRenderingData();
        }
        assert( volume );
        volume_.loadData(
            {
                .resolution = volume->dims,
                .internalFormat = GL_R8,
                .format = GL_RED,
                .type = GL_UNSIGNED_BYTE,
                .filter = params.volumeFilterType
            },
            volume->data );
    }
    else
    {
        volume_.bind();
        setTextureFilterType( params.volumeFilterType, true );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "volume" ), 0 ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_TEXTURE )
    {
        const auto& volume = objVoxels_->vdbVolume();
        std::vector<Color> denseMap( 256 );
        Vector2f realMinMax;
        realMinMax[0] = volume.min;
        realMinMax[1] = volume.max;
        auto size = realMinMax[1] - realMinMax[0];
        int minIndex = int( ( params.min - realMinMax[0] ) / size * denseMap.size() );
        int maxIndex = int( ( params.max - realMinMax[0] ) / size * denseMap.size() );
        auto diff = maxIndex - minIndex;
        auto getAlpha = [&] ( float ratio )->uint8_t
        {
            if ( params.alphaType == ObjectVoxels::VolumeRenderingParams::AlphaType::Constant )
                return params.alphaLimit;
            if ( params.alphaType == ObjectVoxels::VolumeRenderingParams::AlphaType::LinearIncreasing )
                return uint8_t( ratio * float( params.alphaLimit ) );
            if ( params.alphaType == ObjectVoxels::VolumeRenderingParams::AlphaType::LinearDecreasing )
                return uint8_t( ( 1.0f - ratio ) * float( params.alphaLimit ) );
            return 0;
        };
        if ( params.lutType == ObjectVoxels::VolumeRenderingParams::LutType::GrayShades )
        {
            for ( int i = 0; i < denseMap.size(); ++i )
            {
                bool zeroAlpha = i < minIndex || i > maxIndex;
                float ratio = std::clamp( float( i - minIndex ) / float( diff ), 0.0f, 1.0f );
                uint8_t color = uint8_t( ratio * 255.0f );
                denseMap[i] = Color( color, color, color, zeroAlpha ? 0 : getAlpha( ratio ) );
            }
        }
        else if ( params.lutType == ObjectVoxels::VolumeRenderingParams::LutType::Rainbow )
        {
            constexpr std::array<Color, 7> rainbow{
                Color::red(),
                Color( 255,127,0 ),
                Color::yellow(),
                Color::green(),
                Color::blue(),
                Color( 75,0,130 ),
                Color( 148,0,211 )
            };
            for ( int i = 0; i < denseMap.size(); ++i )
            {
                bool zeroAlpha = i < minIndex || i > maxIndex || diff <= 0;
                if ( zeroAlpha )
                {
                    denseMap[i] = Color( 0, 0, 0, 0 );
                    continue;
                }
                float ratio = std::clamp( float( i - minIndex ) / float( diff ), 0.0f, 1.0f );
                float colorRatio = ratio * float( rainbow.size() );
                const auto& startColor = rainbow[int( std::floor( colorRatio ) )];
                const auto& stopColor = rainbow[int( std::ceil( colorRatio ) )];
                colorRatio = colorRatio - std::floor( colorRatio );

                denseMap[i] = startColor * ( 1.0f - colorRatio ) + stopColor * colorRatio;
                denseMap[i].a = getAlpha( ratio );
            }
        }
        else
        {
            for ( int i = 0; i < denseMap.size(); ++i )
            {
                bool zeroAlpha = i < minIndex || i > maxIndex;
                float ratio = std::clamp( float( i - minIndex ) / float( diff ), 0.0f, 1.0f );
                denseMap[i] = params.oneColor;
                denseMap[i].a = zeroAlpha ? 0 : getAlpha( ratio );
            }
        }
        denseMap_.loadData(
            { .resolution = Vector2i( denseMap.size(),1 ), .internalFormat = GL_RGBA8, .format = GL_RGBA, .type = GL_UNSIGNED_BYTE },
            denseMap );
    }
    else
        denseMap_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "denseMap" ), 1 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "shadingMode" ), int( params.shadingType ) ) );

    dirty_ &= ~( DIRTY_PRIMITIVES | DIRTY_TEXTURE );
}

void RenderVolumeObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &volumeArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );

    dirty_ = DIRTY_PRIMITIVES | DIRTY_TEXTURE;
}

void RenderVolumeObject::freeBuffers_()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &volumeArrayObjId_ ) );
}

void RenderVolumeObject::update_()
{
    dirty_ |= objVoxels_->getDirtyFlags();
    objVoxels_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectVoxels, RenderVolumeObject )

}
#endif