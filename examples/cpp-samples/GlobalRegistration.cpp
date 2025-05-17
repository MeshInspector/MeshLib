#include <MRMesh/MRMultiwayICP.h>

int main( int argc, char* argv[] )
{
    MR::ICPObjects objects;

//! [0]        
    MR::MultiwayICP icp( objects, {
        // set sampling voxel size
        .samplingVoxelSize = 1e-3f,
    } );

    icp.setParams( {} );

    // gather statistics
    icp.updateAllPointPairs();

    // compute new transformations
    auto xfs = icp.calculateTransformations();

    for ( auto i = MR::ObjId( 0 ); i < objects.size(); ++i )
        objects[i].xf = xfs[i];
//! [0]        

    return EXIT_SUCCESS;
}