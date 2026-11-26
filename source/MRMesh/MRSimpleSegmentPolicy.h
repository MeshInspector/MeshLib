#pragma once

#include "MRSegmentMesh.h"
#include "MRMesh.h"
#include <algorithm>

namespace MR
{

/// simple data of one segment for MeshSegmenter: its area
struct SimpleSegmentData
{
    double area = 0;
    SimpleSegmentData& operator +=( const SimpleSegmentData& r ) { area += r.area; return * this; }
};

/// simple data of the common boundary of two segments for MeshSegmenter
struct SimpleSegmentBdData
{
    /// the length of the boundary
    double length = 0;
    /// the integral of some curvature metric along the boundary
    double lengthCurv = 0;
    SimpleSegmentBdData& operator +=( const SimpleSegmentBdData& r ) { length += r.length; lengthCurv += r.lengthCurv; return * this; }
};

/// simple policy for MeshSegmenter, which prefers merging
/// smaller by area segments, with long common boundary, and with low average curvature metric on it
struct SimpleSegmentPolicy
{
    const Mesh& mesh;
    const EdgeMetric& curvMetric;

    using SegmentData = SimpleSegmentData;
    using SegmentBdData = SimpleSegmentBdData;

    SegmentData segmentFromTri( FaceId f ) const { return { .area = mesh.area( f ) }; }

    SegmentBdData bdFromEdge( UndirectedEdgeId ue ) const
    {
        const auto len = mesh.edgeLength( ue );
        return { .length = len, .lengthCurv = len * curvMetric( ue ) };
    }

    static double mergePenalty( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd )
    {
        if ( bd.length <= 0 )
            return prohibitMergePenalty;
        return std::min( a.area, b.area ) * bd.lengthCurv / ( bd.length * bd.length );
    }
};

} //namespace MR
