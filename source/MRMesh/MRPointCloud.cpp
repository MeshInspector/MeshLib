#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRComputeBoundingBox.h"
#include "MRPch/MRSpdlog.h"
#include "MRPlane3.h"
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

void PointCloud::addPartByMask( const PointCloud& from, const VertBitSet& fromVerts, VertMap* oldToNewMap /*= nullptr*/ )
{
    const auto& fromPoints = from.points;
    const auto& fromNormals = from.normals;

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
        oldToNewMap->resize( fromValidVerts.count() );
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
    for ( auto& p : points )
    {
        p += 2.0f * ( plane.project( p ) - p );
    }

    invalidateCaches();
}

} //namespace MR
