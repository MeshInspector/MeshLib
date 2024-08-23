#include "MRMarkedVoxelSlice.h"

#ifndef MRVIEWER_NO_VOXELS
#include <MRVoxels/MRObjectVoxels.h>
#include <MRVoxels/MRVDBFloatGrid.h>

namespace MR
{

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
    auto dimXY = dimX * size_t( dims_.y );

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

        Color c;
        auto val = accessor.getValue( coord );
        float normedValue = ( val - params_.min ) / ( params_.max - params_.min );
        c = Color( Vector3f::diagonal( normedValue ) );
        if ( params_.inactiveVoxelColor && !accessor.isValueOn( coord ) )
            c = blend( *params_.inactiveVoxelColor, c );

        VoxelId voxelIndex = VoxelId( coord[0] + coord[1] * dimX + coord[2] * dimXY );

        for ( const auto& backMark : params_.customBackgroundMarks )
            if ( backMark.mask.test( voxelIndex ) )
                c = blend( c, backMark.color );

        if ( params_.marks[MaskType::Segment].mask.test( voxelIndex ) )
            c = blend( params_.marks[MaskType::Segment].color, c );

        if ( params_.marks[MaskType::Inside].mask.test( voxelIndex ) )
            c = blend( params_.marks[MaskType::Inside].color, c );
        else if ( params_.marks[MaskType::Outside].mask.test( voxelIndex ) )
            c = blend( params_.marks[MaskType::Outside].color, c );

        for ( const auto& foreMark : params_.customForegroundMarks )
            if ( foreMark.mask.test( voxelIndex ) )
                c = blend( foreMark.color, c );

        texture[i] = c;
    }
    update( { { texture, { textureWidth , textureHeight } } } );
}

} //namespace MR
#endif
