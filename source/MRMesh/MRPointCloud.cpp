#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRComputeBoundingBox.h"
#include "MRPch/MRSpdlog.h"
#include "MRPlane3.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Box3f PointCloud::getBoundingBox() const
{ 
    return getAABBTree().getBoundingBox();
}

Box3f PointCloud::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, validPoints, toWorld );
}

void PointCloud::addPartByMask( const PointCloud& from, const VertBitSet& fromVerts, VertMap* oldToNewMap, const VertNormals * extNormals )
{
    MR_TIMER
    const auto& fromPoints = from.points;
    const auto& fromNormals = extNormals ? *extNormals : from.normals;

    const bool consistentNormals = normals.size() == 0 || ( points.size() == normals.size() && fromPoints.size() == fromNormals.size() );
    assert( consistentNormals );
    if ( !consistentNormals )
        return;

    const bool useNormals = points.size() == normals.size() && fromPoints.size() == fromNormals.size();

    VertBitSet fromValidVerts = fromVerts & from.validPoints;
    VertId idIt = VertId( points.size() );
    const auto newSize = points.size() + fromValidVerts.count();
    points.resize( newSize );
    validPoints.resize( newSize, true );
    if ( useNormals )
        normals.resize( newSize );
    if ( oldToNewMap )
        oldToNewMap->resize( fromValidVerts.find_last() + 1 );
    for ( auto v : fromValidVerts )
    {
        points[idIt] = fromPoints[v];
        if ( useNormals )
            normals[idIt] = fromNormals[v];
        if ( oldToNewMap )
            ( *oldToNewMap )[v] = idIt;
        idIt++;
    }

    invalidateCaches();
}

VertId PointCloud::addPoint(const Vector3f& point)
{
    VertId id(points.size());
    points.push_back(point);
    validPoints.autoResizeSet(id);

    if ( !normals.empty() )
    {
        spdlog::warn( "Trying to add point without normal to oriented point cloud, adding empty normal" );
        normals.emplace_back();
        assert( normals.size() == points.size() );
    }
    return id;
}

VertId PointCloud::addPoint(const Vector3f& point, const Vector3f& normal)
{
    assert( normals.size() == points.size() );

    VertId id(points.size());
    points.push_back(point);
    validPoints.autoResizeSet(id);
    normals.push_back(normal);
    return id;
}

const AABBTreePoints& PointCloud::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePoints( *this ); } );
}

size_t PointCloud::heapBytes() const
{
    return points.heapBytes()
        + normals.heapBytes()
        + validPoints.heapBytes()
        + AABBTreeOwner_.heapBytes();
}

void PointCloud::mirror( const Plane3f& plane )
{
    MR_TIMER
    BitSetParallelFor( validPoints, [&] ( VertId id )
    {
        points[id] += 2.0f * ( plane.project( points[id] ) - points[id] );
        if ( !normals.empty() )
            normals[id] -= 2.0f * dot( normals[id], plane.n ) * plane.n;
    } );

    invalidateCaches();
}

bool PointCloud::pack( VertMap * outNew2Old )
{
    MR_TIMER
    const auto newSz = validPoints.count();
    if ( points.size() == newSz )
    {
        assert( normals.empty() || normals.size() == newSz );
        assert( validPoints.size() == newSz );
        return false;
    }

    if ( outNew2Old )
    {
        outNew2Old->clear();
        outNew2Old->reserve( newSz );
    }

    VertCoords packedPoints;
    packedPoints.reserve( newSz );
    VertNormals packedNormals;
    if ( !normals.empty() )
        packedNormals.reserve( newSz );

    for ( auto v : validPoints )
    {
        packedPoints.push_back( points[v] );
        if ( !normals.empty() )
            packedNormals.push_back( normals[v] );
        if ( outNew2Old )
            outNew2Old->push_back( v );
    }

    assert( packedPoints.size() == newSz );
    assert( packedNormals.empty() || packedNormals.size() == newSz );
    points = std::move( packedPoints );
    normals = std::move( packedNormals );
    validPoints.clear();
    validPoints.resize( newSz, true );

    invalidateCaches();
    return true;
}

VertBMap PointCloud::packOptimally()
{
    MR_TIMER

    getAABBTree(); // ensure that tree is constructed
    VertBMap map;
    map.b.resize( points.size() );
    const bool packed = validPoints.count() == points.size();
    if ( !packed )
    {
        for ( VertId v = 0_v; v < map.b.size(); ++v )
            if ( !validPoints.test( v ) )
                map.b[v] = VertId{};
    }
    AABBTreeOwner_.get()->getLeafOrderAndReset( map );

    VertCoords newPoints;
    newPoints.resizeNoInit( map.tsize );
    VertNormals newNormals;
    if ( hasNormals() )
        newNormals.resizeNoInit( map.tsize );

    ParallelFor( 0_v, map.b.endId(), [&]( VertId oldv )
    {
        auto newv = map.b[oldv];
        if ( !newv )
            return;
        newPoints[newv] = points[oldv];
        if ( hasNormals() )
            newNormals[newv] = normals[oldv];
    } );
    points = std::move( newPoints );
    normals = std::move( newNormals );
    validPoints = {};
    validPoints.resize( points.size(), true );
    return map;
}

} //namespace MR
