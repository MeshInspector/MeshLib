#include "MRObjectPointsHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRChangePointCloudAction.h>
#include <MRMesh/MRChangeVertsColorMapAction.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTimer.h>
#include <MRMesh/MRBitSetParallelFor.h>

namespace MR
{

bool packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints )
{
    MR_TIMER

    if ( !objPoints || !objPoints->pointCloud() )
        return false;

    Historian<ChangePointCloudAction> h( "set cloud", objPoints );
        
    VertMap new2Old;
    if ( !objPoints->varPointCloud()->pack( &new2Old ) )
    {
        h.cancelAction();
        return false;
    }

    if ( !objPoints->getVertsColorMap().empty() )
    {
        Historian<ChangeVertsColorMapAction> hCM( "color map update", objPoints );
        VertColors colors;
        objPoints->updateVertsColorMap( colors );
        for ( VertId n = 0_v; n < new2Old.size(); ++n )
        {
            VertId o = new2Old[n];
            assert( n <= o );
            colors[n] = colors[o];
        }
        colors.resize( new2Old.size() );
        objPoints->updateVertsColorMap( colors );
    }

    // update faces in the selection
    const auto & oldSel = objPoints->getSelectedPoints();
    if ( oldSel.any() )
    {
        Historian<ChangePointPointSelectionAction> hs( "selection", objPoints );
        VertBitSet newSel( new2Old.size() );
        BitSetParallelForAll( newSel, [&]( VertId n )
        {
            if ( oldSel.test( new2Old[n] ) )
                newSel.set( n );
        } );
        objPoints->selectPoints( std::move( newSel ) );
    }

    return true;
}

bool packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, VertBitSet newValidVerts )
{
    if ( !objPoints || !objPoints->pointCloud() )
        return false;
    objPoints->varPointCloud()->validPoints = std::move( newValidVerts );
    return packPointsWithHistory( objPoints );
}

} //namespace MR
