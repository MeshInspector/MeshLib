#include "MRRenderVolumeObject.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRGLStaticHolder.h"
#include "MRMesh/MRFloatGrid.h"

namespace
{
using namespace MR;
using SimpleVolumeU8 = VoxelsVolume<std::vector<uint8_t>>;

SimpleVolumeU8 vdbVolumeToNormedSimpleVolume( const VdbVolume& vdbVolume )
{
    if ( !vdbVolume.data )
        return {};
    SimpleVolumeU8 res;
    res.max = vdbVolume.max;
    res.min = vdbVolume.min;
    res.voxelSize = vdbVolume.voxelSize;
    auto activeBox = vdbVolume.data->evalActiveVoxelBoundingBox();
    res.dims = Vector3i( activeBox.dim().x(), activeBox.dim().y(), activeBox.dim().z() );
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto accessor = vdbVolume.data->getConstAccessor();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto coord = indexer.toPos( VoxelId( i ) );
            auto vdbCoord = openvdb::Coord( coord.x + activeBox.min().x(), coord.y + activeBox.min().y(), coord.z + activeBox.min().z() );
            res.data[i] = uint8_t( std::clamp( ( accessor.getValue( vdbCoord ) - res.min ) / ( res.max - res.min ), 0.0f, 1.0f ) * 255.0f );
        }
    } );
    return res;
}
}

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
    if ( !getViewerInstance().isGLInitialized() )
    {
        objVoxels_->resetDirty();
        return;
    }
    update_();

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

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

    bindVolume_();
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrixPtr ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrixPtr ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), 
        objVoxels_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ), 
        float( renderParams.viewport.x ), float( renderParams.viewport.y ), 
        float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );

    const auto& voxelSize = objVoxels_->vdbVolume().voxelSize;
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "voxelSize" ), voxelSize.x, voxelSize.y, voxelSize.z ) );

    constexpr GLfloat textureQuad[18] =
    {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f
    };
    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, volumeBufferObjId_ ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, textureQuad, GL_STATIC_DRAW ) );

    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );

    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );

    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, static_cast < GLsizei > ( 6 ) ) );
}

void RenderVolumeObject::renderPicker( const BaseRenderParams&, unsigned )
{
    // TODO: picker for volume
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
    bindVolume_();
}

void RenderVolumeObject::bindVolume_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( dirty_ & DIRTY_PRIMITIVES )
    {
        auto volume = vdbVolumeToNormedSimpleVolume( objVoxels_->vdbVolume() );
        volume_.loadData(
            { .resolution = volume.dims, .internalFormat = GL_R8, .format = GL_RED, .type = GL_UNSIGNED_BYTE },
            volume.data );
    }
    else
        volume_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "volume" ), 0 ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_TEXTURE )
    {
        auto volume = objVoxels_->vdbVolume();
        auto isoValue = ( objVoxels_->getIsoValue() - volume.min ) / ( volume.max - volume.min );
        bool dense = volume.data->getGridClass() != openvdb::GRID_LEVEL_SET;
        std::vector<Color> denseMap( 256 );
        for ( int i = 0; i < denseMap.size(); ++i )
            denseMap[255 - i] = Color( i, i, i, 10 );
        int passOver = int( isoValue * denseMap.size() );
        for ( int i = 0; i < denseMap.size(); ++i )
        {
            if ( dense == ( i < passOver ) )
                denseMap[i].a = 0;
        }
        denseMap_.loadData(
            { .resolution = Vector2i( denseMap.size(),1 ), .internalFormat = GL_RGBA8, .format = GL_RGBA, .type = GL_UNSIGNED_BYTE },
            denseMap );
    }
    else
        denseMap_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "denseMap" ), 1 ) );

    dirty_ &= ~( DIRTY_PRIMITIVES | DIRTY_TEXTURE );
}

void RenderVolumeObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &volumeArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );
    GL_EXEC( glGenBuffers( 1, &volumeBufferObjId_ ) );

    dirty_ = DIRTY_PRIMITIVES | DIRTY_TEXTURE;
}

void RenderVolumeObject::freeBuffers_()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &volumeArrayObjId_ ) );
    GL_EXEC( glDeleteBuffers( 1, &volumeBufferObjId_ ) );
}

void RenderVolumeObject::update_()
{
    dirty_ |= objVoxels_->getDirtyFlags();
    objVoxels_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectVoxels, RenderVolumeObject )

}