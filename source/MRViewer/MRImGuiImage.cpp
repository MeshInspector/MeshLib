#include "MRImGuiImage.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRFloatGrid.h"
#include "MRMesh/MRImageSave.h"
#include "MRGLMacro.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

ImGuiImage::ImGuiImage()
{
}

ImGuiImage::~ImGuiImage()
{
}

void ImGuiImage::update( const MeshTexture& texture )
{
    texture_ = texture;
    bind_();
}

void ImGuiImage::bind_()
{
    if ( !getViewerInstance().isGLInitialized() )
        return;
    glTex_.loadData(
        { 
            .resolution = texture_.resolution, 
            .internalFormat = GL_RGBA, 
            .format = GL_RGBA, 
            .type = GL_UNSIGNED_BYTE, 
            .wrap = texture_.wrap, 
            .filter = texture_.filter
        },
        texture_.pixels );
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

void MarkedVoxelSlice::saveCurrentSliceToFile( const std::filesystem::path& path )
{
    ImageSave::toAnySupportedFormat( getMeshTexture(), path );
}

void MarkedVoxelSlice::saveAllSlicesToDirectory( const std::filesystem::path& path )
{
    const auto& box = getActiveBox();
    const auto& v = getActiveVoxel();

    switch ( getActivePlane() )
    {
    case SlicePlain::XY:
    {
        for ( int i = box.min.z; i <= box.max.z; ++i )
        {
            setActiveVoxel( { v.x, v.y, i } );
            ImageSave::toAnySupportedFormat( getMeshTexture(), path.string() + "/slice_" + std::to_string( i ) + ".png" );
        }
        break;
    }
    case SlicePlain::YZ:
    {
        for ( int i = box.min.x; i <= box.max.x; ++i )
        {
            setActiveVoxel( { i, v.y, v.z } );
            ImageSave::toAnySupportedFormat( getMeshTexture(), path.string() + "/slice_" + std::to_string( i ) + ".png" );
        }
        break;
    }
    case SlicePlain::ZX:
    {
        for ( int i = box.min.y; i <= box.max.y; ++i )
        {
            setActiveVoxel( { v.x, i, v.z } );
            ImageSave::toPng( getMeshTexture(), path.string() + "/slice_" + std::to_string( i ) + ".png" );
        }
        break;
    }
    default:
        break;
    }

    setActiveVoxel( v );
}

#endif

}
