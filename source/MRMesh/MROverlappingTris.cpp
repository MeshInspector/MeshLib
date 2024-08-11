#include "MROverlappingTris.h"
#include "MRMeshDistance.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Expected<FaceBitSet> findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings )
{
    MR_TIMER
    FaceBitSet res( mp.mesh.topology.faceSize() );
    if ( BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&]( FaceId f )
    {
        const auto fDirDblArea = mp.mesh.dirDblArea( f );
        const auto fDblArea = fDirDblArea.length();
        const auto fnormal = fDirDblArea.normalized();
        const auto tri = mp.mesh.getTriPoints( f );
        bool overlapping = false;
        auto onNeiTriangle = [&]( const Vector3f &, FaceId f1, const Vector3f &, float /*distSq*/ )
        {
            if ( f == f1 )
                return ProcessOneResult::ContinueProcessing;

            const auto f1DirDblArea = mp.mesh.dirDblArea( f1 );
            const auto f1DblArea = f1DirDblArea.length();
            const auto f1normal = f1DirDblArea.normalized();
            if ( fDblArea * settings.minAreaFraction > f1DblArea )
                return ProcessOneResult::ContinueProcessing;

            if ( dot( fnormal, f1normal ) > settings.maxNormalDot )
                return ProcessOneResult::ContinueProcessing;
            overlapping = true;
            return ProcessOneResult::StopProcessing;
        };
        processCloseTriangles( mp, tri, settings.maxDistSq, onNeiTriangle );
        if ( overlapping )
            res.set( f );
    }, settings.cb ) )
        return res;
    return unexpectedOperationCanceled();
}

} //namespace MR
