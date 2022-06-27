#include "MRImGuiImage.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRFloatGrid.h"
#include "MRGLMacro.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

ImGuiImage::ImGuiImage()
{
    if ( !getViewerInstance().isGLInitialized() )
        return;
    GL_EXEC( glGenTextures( 1, &id_ ) );
    initialized_ = true;
}

ImGuiImage::~ImGuiImage()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;
    if ( initialized_ )
    {
        GL_EXEC( glDeleteTextures( 1, &id_ ) );
    }
}

void ImGuiImage::update( const MeshTexture& texture )
{
    texture_ = texture;
    bind_();
}

void ImGuiImage::bind_()
{
    if ( !initialized_ )
        return;

    GL_EXEC( glBindTexture( GL_TEXTURE_2D, id_ ) );

    int warp{0};
    switch ( texture_.warp )
    {
    case MeshTexture::WarpType::Clamp:
        warp = GL_CLAMP_TO_EDGE;
        break;
    case MeshTexture::WarpType::Repeat:
        warp = GL_REPEAT;
        break;
    case MeshTexture::WarpType::Mirror:
        warp = GL_MIRRORED_REPEAT;
        break;
    default:
        break;
    }

    int filter{0};
    switch ( texture_.filter )
    {
    case MeshTexture::FilterType::Discrete:
        filter = GL_NEAREST;
        break;
    case MeshTexture::FilterType::Linear:
        filter = GL_LINEAR;
        break;
    default:
        break;
    }

    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, warp ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, warp ) );
    GL_EXEC( glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, texture_.resolution.x, texture_.resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_.pixels.data() ) );
}

#ifndef __EMSCRIPTEN__
MarkedVoxelSlice::MarkedVoxelSlice( const ObjectVoxels& voxels )
{
    grid_ = voxels.grid();
    params_.activeBox = voxels.getActiveBounds();
    dims_ = voxels.dimensions();
}

void MarkedVoxelSlice::forceUpdate()
{
    if ( !grid_ )
        return;

    auto dimX = dims_.x;
    auto dimXY = dimX * dims_.y;

    auto activeDims = params_.activeBox.size();

    auto textureWidth = activeDims[( params_.activePlane + 1 ) % 3];
    auto textureHeight = activeDims[( params_.activePlane + 2 ) % 3];

    std::vector<Color> texture( textureWidth * textureHeight );
    int curMain = params_.activeVoxel[params_.activePlane];
    const auto& accessor = grid_->getConstAccessor();
    for ( int i = 0; i < texture.size(); ++i )
    {
        openvdb::Coord coord;
        coord[params_.activePlane] = curMain;
        coord[( params_.activePlane + 1 ) % 3] = ( i % textureWidth ) + params_.activeBox.min[( params_.activePlane + 1 ) % 3];
        coord[( params_.activePlane + 2 ) % 3] = ( i / textureWidth ) + params_.activeBox.min[( params_.activePlane + 2 ) % 3];

        auto val = accessor.getValue( coord );
        float normedValue = ( val - params_.min ) / ( params_.max - params_.min );
        texture[i] = Color( Vector3f::diagonal( normedValue ) );

        VoxelId voxelIndex = VoxelId( coord[0] + coord[1] * dimX + coord[2] * dimXY );

        for ( const auto& backMark : params_.customBackgroundMarks )
            if ( backMark.mask.test( voxelIndex ) )
                texture[i] = backMark.color;

        if ( params_.marks[MaskType::Segment].mask.test( voxelIndex ) )
            texture[i] = params_.marks[MaskType::Segment].color;

        if ( params_.marks[MaskType::Inside].mask.test( voxelIndex ) )
            texture[i] = params_.marks[MaskType::Inside].color;
        else if ( params_.marks[MaskType::Outside].mask.test( voxelIndex ) )
            texture[i] = params_.marks[MaskType::Outside].color;

        for ( const auto& foreMark : params_.customForegroundMarks )
            if ( foreMark.mask.test( voxelIndex ) )
                texture[i] = foreMark.color;
    }
    update( { { texture, { textureWidth , textureHeight } } } );
}

#endif

}
