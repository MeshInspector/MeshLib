#include "MROverlappingTris.h"
#include "MRMeshDistance.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

/// returns true if the projections of two triangles on the plane with normal (n) overlap deeper than a tiny fraction of their sizes,
/// the tolerance rejects the triangles only touching one another (e.g. sharing a vertex) despite rounding errors
bool doProjectionsOverlap( const Triangle3f & a, const Triangle3f & b, const Vector3f & n )
{
    auto separatedBy = [&]( const Triangle3f & t )
    {
        for ( int i = 0; i < 3; ++i )
        {
            const auto axis = cross( n, t[( i + 1 ) % 3] - t[i] );
            float minA = FLT_MAX, maxA = -FLT_MAX, minB = FLT_MAX, maxB = -FLT_MAX;
            for ( int j = 0; j < 3; ++j )
            {
                const auto pa = dot( axis, a[j] - t[i] );
                minA = std::min( minA, pa );
                maxA = std::max( maxA, pa );
                const auto pb = dot( axis, b[j] - t[i] );
                minB = std::min( minB, pb );
                maxB = std::max( maxB, pb );
            }
            const auto tol = 1e-3f * std::min( maxA - minA, maxB - minB );
            if ( maxA <= minB + tol || maxB <= minA + tol )
                return true;
        }
        return false;
    };
    return !separatedBy( a ) && !separatedBy( b );
}

} //anonymous namespace

Expected<FaceBitSet> findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings )
{
    MR_TIMER;
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

            const auto d = dot( fnormal, f1normal );
            if ( d > settings.maxNormalDot )
            {
                if ( d < settings.minNormalDot || mp.mesh.topology.sharedEdge( f, f1 )
                    || !doProjectionsOverlap( tri, mp.mesh.getTriPoints( f1 ), fnormal + f1normal ) )
                    return ProcessOneResult::ContinueProcessing;
            }

            if ( settings.pred && !settings.pred( f, f1 ) )
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
