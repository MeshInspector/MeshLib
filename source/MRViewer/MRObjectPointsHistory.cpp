#include "MRObjectPointsHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRChangePointCloudAction.h>
#include <MRMesh/MRChangeVertsColorMapAction.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTimer.h>
#include <MRMesh/MRParallelFor.h>
#include <MRMesh/MRBitSetParallelFor.h>
#include <MRMesh/MRBuffer.h>

namespace MR
{

static void packPointsWithHistoryCore( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reorder, VertBitSet * newValidVerts )
{
    MR_TIMER

    if ( !objPoints || !objPoints->pointCloud() )
        return;

    Historian<ChangePointCloudAction> h( "set cloud", objPoints );

    if ( newValidVerts )
    {
        objPoints->varPointCloud()->validPoints = std::move( *newValidVerts );
        objPoints->varPointCloud()->invalidateCaches();
    }

    const auto map = objPoints->varPointCloud()->pack( reorder );

    if ( !objPoints->getVertsColorMap().empty() )
    {
        Historian<ChangeVertsColorMapAction> hCM( "color map update", objPoints );
        VertColors newColors;
        newColors.resizeNoInit( map.tsize );
        const auto & oldColors = objPoints->getVertsColorMap();
        ParallelFor( 0_v, map.b.endId(), [&] ( VertId oldv )
        {
            auto newv = map.b[oldv];
            if ( !newv )
                return;
            newColors[newv] = oldColors[oldv];
        } );
        objPoints->setVertsColorMap( std::move( newColors ) );
    }

    // update faces in the selection
    const auto & oldSel = objPoints->getSelectedPoints();
    if ( oldSel.any() )
    {
        Historian<ChangePointPointSelectionAction> hs( "selection", objPoints );
        VertBitSet newSel( map.tsize );
        for ( auto oldv : oldSel )
            if ( auto newv = map.b[ oldv ] )
                newSel.set( newv );
        objPoints->selectPoints( std::move( newSel ) );
    }
}

void packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reorder )
{
    packPointsWithHistoryCore( objPoints, reorder, nullptr );
}

void packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reorder, VertBitSet newValidVerts )
{
    packPointsWithHistoryCore( objPoints, reorder, &newValidVerts );
}

} //namespace MR
