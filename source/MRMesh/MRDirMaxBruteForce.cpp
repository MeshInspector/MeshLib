#include "MRDirMaxBruteForce.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include <MRPch/MRTBB.h>
#include <cfloat>
#include <compare>

namespace MR
{

namespace
{

struct ProjectedVertex
{
    float proj = -FLT_MAX; //projection of points[v] on the direction of interest
    VertId v;
    auto operator <=>( const ProjectedVertex& ) const = default;
};

template<class V>
VertId findDirMaxBruteForceT( const V & dir, const Vector<V, VertId> & points, const VertBitSet * region )
{
    MR_TIMER;

    auto pv = parallel_reduce( tbb::blocked_range( 0_v, points.endId(), 1024 ), ProjectedVertex{},
        [&] ( const auto & range, ProjectedVertex curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
            {
                if ( !contains( region, v ) )
                    continue;
                curr = std::max( curr, ProjectedVertex{ dot( points[v], dir ), v } );
            }
            return curr;
        },
        [] ( ProjectedVertex a, const ProjectedVertex & b ) { a = std::max( a, b ); return a; }
    );
    return pv.v;
}

template<class V>
MinMaxArg<float, VertId> findDirMinMaxBruteForceT( const V & dir, const Vector<V, VertId> & points, const VertBitSet * region )
{
    MR_TIMER;

    return parallel_reduce( tbb::blocked_range( 0_v, points.endId(), 1024 ), MinMaxArg<float, VertId>{},
        [&] ( const auto & range, MinMaxArg<float, VertId> curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
            {
                if ( !contains( region, v ) )
                    continue;
                curr.include( dot( points[v], dir ), v );
            }
            return curr;
        },
        [] ( MinMaxArg<float, VertId> a, const MinMaxArg<float, VertId> & b ) { a.include( b ); return a; }
    );
}

} // anonymous namespace

VertId findDirMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region )
{
    return findDirMaxBruteForceT( dir, points, region );
}

VertId findDirMaxBruteForce( const Vector2f & dir, const VertCoords2 & points, const VertBitSet * region )
{
    return findDirMaxBruteForceT( dir, points, region );
}

VertId findDirMaxBruteForce( const Vector3f & dir, const PointCloud & cloud )
{
    return findDirMaxBruteForce( dir, cloud.points, &cloud.validPoints );
}

VertId findDirMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline )
{
    return findDirMaxBruteForce( dir, polyline.points, &polyline.topology.getValidVerts() );
}

VertId findDirMaxBruteForce( const Vector2f & dir, const Polyline2 & polyline )
{
    return findDirMaxBruteForce( dir, polyline.points, &polyline.topology.getValidVerts() );
}

VertId findDirMaxBruteForce( const Vector3f & dir, const MeshPart & mp )
{
    if ( !mp.region )
        return findDirMaxBruteForce( dir, mp.mesh.points, &mp.mesh.topology.getValidVerts() );

    MR_TIMER;
    auto pv = parallel_reduce( tbb::blocked_range( 0_f, FaceId( mp.mesh.topology.faceSize() ), 1024 ), ProjectedVertex{},
        [&] ( const auto & range, ProjectedVertex curr )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mp.region->test( f ) )
                    continue;
                VertId vs[3];
                mp.mesh.topology.getTriVerts( f, vs );
                for ( auto v : vs )
                    curr = std::max( curr, ProjectedVertex{ dot( mp.mesh.points[v], dir ), v } );
            }
            return curr;
        },
        [] ( ProjectedVertex a, const ProjectedVertex & b ) { a = std::max( a, b ); return a; }
    );
    return pv.v;
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region )
{
    return findDirMinMaxBruteForceT( dir, points, region );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector2f & dir, const VertCoords2 & points, const VertBitSet * region )
{
    return findDirMinMaxBruteForceT( dir, points, region );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const PointCloud & cloud )
{
    return findDirMinMaxBruteForce( dir, cloud.points, &cloud.validPoints );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline )
{
    return findDirMinMaxBruteForce( dir, polyline.points, &polyline.topology.getValidVerts() );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector2f & dir, const Polyline2 & polyline )
{
    return findDirMinMaxBruteForce( dir, polyline.points, &polyline.topology.getValidVerts() );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const MeshPart & mp )
{
    if ( !mp.region )
        return findDirMinMaxBruteForce( dir, mp.mesh.points, &mp.mesh.topology.getValidVerts() );

    MR_TIMER;
    return parallel_reduce( tbb::blocked_range( 0_f, FaceId( mp.mesh.topology.faceSize() ), 1024 ), MinMaxArg<float, VertId>{},
        [&] ( const auto & range, MinMaxArg<float, VertId> curr )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mp.region->test( f ) )
                    continue;
                VertId vs[3];
                mp.mesh.topology.getTriVerts( f, vs );
                for ( auto v : vs )
                    curr.include( dot( mp.mesh.points[v], dir ), v );
            }
            return curr;
        },
        [] ( MinMaxArg<float, VertId> a, const MinMaxArg<float, VertId> & b ) { a.include( b ); return a; }
    );
}

} //namespace MR
