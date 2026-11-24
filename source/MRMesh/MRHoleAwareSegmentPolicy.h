#pragma once

#include "MRSimpleSegmentPolicy.h"
#include "MRRingIterator.h"

namespace MR
{

/// data of one segment for MeshSegmenter: its area and the length of its boundary with mesh holes
struct HoleAwareSegmentData : SimpleSegmentData
{
    /// the length of segment's boundary with mesh holes
    double holeBdLength = 0;
    HoleAwareSegmentData& operator +=( const HoleAwareSegmentData& r )
    {
        SimpleSegmentData::operator +=( r );
        holeBdLength += r.holeBdLength;
        return * this;
    }
};

/// policy for MeshSegmenter with the same preferences as in SimpleSegmentPolicy, which additionally
/// prohibits merging of two segments over a short common boundary if one of them has longer boundary with mesh holes
struct HoleAwareSegmentPolicy
{
    const Mesh& mesh;
    const EdgeMetric& curvMetric;

    /// two segments are not merged if their common boundary is shorter than this value,
    /// and at least one of them has the boundary with mesh holes longer than their common boundary
    double shortBdLength = 0;

    using SegmentData = HoleAwareSegmentData;
    using SegmentBdData = SimpleSegmentBdData;

    SegmentData segmentFromTri( FaceId f ) const
    {
        SegmentData res;
        res.area = mesh.area( f );
        for ( auto e : leftRing( mesh.topology, f ) )
            if ( !mesh.topology.right( e ) )
                res.holeBdLength += mesh.edgeLength( e );
        return res;
    }

    SegmentBdData bdFromEdge( UndirectedEdgeId ue ) const { return SimpleSegmentPolicy{ mesh, curvMetric }.bdFromEdge( ue ); }

    double mergePenalty( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd ) const
    {
        if ( bd.length < shortBdLength && std::max( a.holeBdLength, b.holeBdLength ) > bd.length )
            return prohibitMergePenalty;
        return SimpleSegmentPolicy::mergePenalty( a, b, bd );
    }
};

} //namespace MR
