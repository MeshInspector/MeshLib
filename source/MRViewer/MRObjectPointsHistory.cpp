#include "MRObjectPointsHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRChangePointCloudAction.h>
#include <MRMesh/MRChangeVertsColorMapAction.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTimer.h>

namespace MR
{

static void packPointsWithHistoryCore( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reorder, VertBitSet * newValidVerts )
{
    MR_TIMER

    if ( !objPoints || !objPoints->pointCloud() )
        return;

    auto packed = pack( *objPoints, reorder, newValidVerts );

    {
        Historian<ChangePointCloudAction> h( "set cloud", objPoints );
        std::shared_ptr<PointCloud> tmp;
        packed->swapPointCloud( tmp );    // tmp := packed
        objPoints->swapPointCloud( tmp ); // objPoints := tmp
    }

    {
        Historian<ChangeVertsColorMapAction> hCM( "color map update", objPoints );
        VertColors tmp;
        packed->updateVertsColorMap( tmp );    // tmp := packed
        objPoints->updateVertsColorMap( tmp ); // objPoints := tmp
    }

    {
        Historian<ChangePointPointSelectionAction> hs( "selection", objPoints );
        VertBitSet tmp;
        packed->updateSelectedPoints( tmp );    // tmp := packed
        objPoints->updateSelectedPoints( tmp ); // objPoints := tmp
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
