#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRComputeBoundingBox.h"
#include "MRPlane3.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

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

Box3f PointCloud::computeBoundingBox( const VertBitSet * region, const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, getVertIds( region ), toWorld );
}

Vector3f PointCloud::findCenterFromPoints() const
{
    MR_TIMER;
    const auto num = calcNumValidPoints();
    if ( num <= 0 )
    {
        assert( false );
        return {};
    }
    auto sumPos = parallel_deterministic_reduce( tbb::blocked_range( 0_v, VertId{ points.size() }, 1024 ), Vector3d{},
    [&] ( const auto & range, Vector3d curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( validPoints.test( v ) )
                curr += Vector3d{ points[v] };
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
    return Vector3f{ sumPos / (double)num };
}

Vector3f PointCloud::findCenterFromBBox() const
{
    return computeBoundingBox().center();
}

void PointCloud::addPartByMask( const PointCloud& from, const VertBitSet& fromVerts, const CloudPartMapping& outMap, const VertNormals * extNormals )
{
    MR_TIMER;
    const auto& fromPoints = from.points;
    const auto& fromNormals = extNormals ? *extNormals : from.normals;

    const bool useNormals = hasNormals() && fromNormals.size() >= fromPoints.size();
    const bool consistentNormals = normals.size() == 0 || useNormals;
    assert( consistentNormals );
    if ( !consistentNormals )
        return;

    VertBitSet fromValidVerts = fromVerts & from.validPoints;
    VertId idIt = VertId( points.size() );
    const auto newSize = points.size() + fromValidVerts.count();
    points.resizeNoInit( newSize );
    validPoints.resize( newSize, true );
    if ( useNormals )
        normals.resize( newSize );
    if ( outMap.src2tgtVerts )
        outMap.src2tgtVerts->resize( fromValidVerts.find_last() + 1 );
    if ( outMap.tgt2srcVerts )
        outMap.tgt2srcVerts->resizeNoInit( points.size() );
    for ( auto v : fromValidVerts )
    {
        points[idIt] = fromPoints[v];
        if ( useNormals )
            normals[idIt] = fromNormals[v];
        if ( outMap.src2tgtVerts )
            ( *outMap.src2tgtVerts )[v] = idIt;
        if ( outMap.tgt2srcVerts )
            ( *outMap.tgt2srcVerts )[idIt] = v;
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
    MR_TIMER;
    BitSetParallelFor( validPoints, [&] ( VertId id )
    {
        points[id] += 2.0f * ( plane.project( points[id] ) - points[id] );
        if ( !normals.empty() )
            normals[id] -= 2.0f * dot( normals[id], plane.n ) * plane.n;
    } );

    invalidateCaches();
}

void PointCloud::flipOrientation( const VertBitSet * region )
{
    MR_TIMER;
    BitSetParallelFor( getVertIds( region ), [&] ( VertId id )
    {
        if ( id < normals.size() )
            normals[id] = -normals[id];
    } );
}

bool PointCloud::pack( VertMap * outNew2Old )
{
    MR_TIMER;
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

std::vector<VertId> PointCloud::getLexicographicalOrder() const
{
    MR_TIMER;
    std::vector<VertId> lexyOrder;
    lexyOrder.reserve( validPoints.count() );
    for ( auto v : validPoints )
        lexyOrder.push_back( v );
    tbb::parallel_sort( lexyOrder.begin(), lexyOrder.end(), [&] ( VertId l, VertId r )
    {
        const auto& ptL = points[l];
        const auto& ptR = points[r];
        return std::tuple{ ptL.x, ptL.y, ptL.z } < std::tuple{ ptR.x, ptR.y, ptR.z };
    } );
    return lexyOrder;
}

VertBMap PointCloud::pack( Reorder reoder )
{
    MR_TIMER;

    const auto numValidPoints = validPoints.count();
    const bool wasPacked = numValidPoints == points.size();

    VertBMap map;
    map.b.resize( points.size() );
    map.tsize = numValidPoints;

    switch ( reoder )
    {
    default:
        assert( false );
        [[fallthrough]];
    case Reorder::None:
    {
        invalidateCaches();
        VertId newId( 0 );
        for ( VertId v = 0_v; v < map.b.size(); ++v )
            if ( validPoints.test( v ) )
                map.b[v] = newId++;
            else
                map.b[v] = VertId{};
        assert( newId == map.tsize );
        break;
    }
    case Reorder::Lexicographically:
    {
        invalidateCaches();
        std::vector<VertId> lexyOrder = getLexicographicalOrder();
        ParallelFor( size_t(0), lexyOrder.size(), [&]( size_t i )
        {
            VertId oldId = lexyOrder[i];
            VertId newId( i );
            map.b[oldId] = newId;
        } );
        if ( !wasPacked )
        {
            ParallelFor( 0_v, map.b.endId(), [&]( VertId v )
            {
                if ( !validPoints.test( v ) )
                    map.b[v] = VertId{};
            } );
        }
        break;
    }
    case Reorder::AABBTree:
    {
        getAABBTree(); // ensure that tree is constructed
        AABBTreeOwner_.update( [&map]( AABBTreePoints& t ) { t.getLeafOrderAndReset( map ); } );
        if ( !wasPacked )
        {
            ParallelFor( 0_v, map.b.endId(), [&]( VertId v )
            {
                if ( !validPoints.test( v ) )
                    map.b[v] = VertId{};
            } );
        }
        break;
    }
    }

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
