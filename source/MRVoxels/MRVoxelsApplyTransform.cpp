#include "MRVoxelsApplyTransform.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRMesh/MRMatrix4.h"
#include "MRVDBConversions.h"
#include "MRObjectVoxels.h"

#include <openvdb/math/Maps.h>
#include <openvdb/tools/GridTransformer.h>


namespace MR
{


VdbVolume transformVdbVolume( const VdbVolume& volume, const AffineXf3f& xf0, bool fixBox, const Box3f& box )
{
    AffineXf3f xf = xf0;
    Box3f newBox;
    {
        Box3f pseudoBox;
        if ( box.valid() )
            pseudoBox = box;
        else
            pseudoBox = Box3f{ { 0.f, 0.f, 0.f }, mult( Vector3f( volume.dims ), volume.voxelSize ) };

        for ( auto c : getCorners( pseudoBox ) )
            newBox.include( xf( c ) );
    }

    if ( fixBox && box.valid() &&
         std::any_of( begin( newBox.min ), end( newBox.min ), [] ( float f ) { return f < 0; } ) )
    {
        Vector3f fixer;
        for ( int i = 0; i < 3; ++i )
        {
            if ( newBox.min[i] < 0 )
            {
                fixer[i] = -newBox.min[i];
                newBox.min[i] = 0;
                newBox.max[i] += fixer[i];
            }
        }

        xf = AffineXf3f::translation( fixer ) * xf;
    }

    Matrix4d m( Matrix3d( xf.A ), Vector3d( div( xf.b, volume.voxelSize ) ) );
    m = m.transposed();
    openvdb::math::AffineMap map( openvdb::math::Mat4d{ ( double* )&m } );
    openvdb::tools::GridTransformer tr( map.getConstMat4() );

    FloatGrid grid = std::make_shared<OpenVdbFloatGrid>();
    tr.transformGrid<openvdb::tools::QuadraticSampler, openvdb::FloatGrid>( *volume.data, *grid );

    VdbVolume res;
    res = volume;
    res.data = grid;
    res.dims = Vector3i( div( newBox.max, volume.voxelSize ) );
    evalGridMinMax( grid, res.min, res.max );

    return res;
}

void voxelsApplyTransform( ObjectVoxels& obj, const AffineXf3f& xf, bool fixBox )
{
    auto r = transformVdbVolume( obj.vdbVolume(), xf, fixBox, obj.getBoundingBox() );
    obj.updateVdbVolume( r );
    obj.updateHistogramAndSurface();
}


}
