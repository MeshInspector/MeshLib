#include "MRDirMaxBruteForce.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRVector2.h"
#include <MRPch/MRTBB.h>
#include <cfloat>

namespace MR
{

namespace
{

template<class V>
VertId findDirMaxBruteForceT( const V & dir, const Vector<V, VertId> & points, const VertBitSet * region )
{
    MR_TIMER;

    auto pv = parallel_reduce( tbb::blocked_range( 0_v, points.endId(), 1024 ), MaxArg<float, VertId>{},
        [&] ( const auto & range, MaxArg<float, VertId> curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
            {
                if ( !contains( region, v ) )
                    continue;
                curr.include( dot( points[v], dir ), v );
            }
            return curr;
        },
        [] ( MaxArg<float, VertId> a, const MaxArg<float, VertId> & b ) { a.include( b ); return a; }
    );
    return pv.arg;
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

VertId findDirMaxBruteForce( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region )
{
    return findDirMaxBruteForce( dir, cloud.points, region ? region : &cloud.validPoints );
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
    auto pv = parallel_reduce( tbb::blocked_range( 0_f, FaceId( mp.mesh.topology.faceSize() ), 1024 ), MaxArg<float, VertId>{},
        [&] ( const auto & range, MaxArg<float, VertId> curr )
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
        [] ( MaxArg<float, VertId> a, const MaxArg<float, VertId> & b ) { a.include( b ); return a; }
    );
    return pv.arg;
}

VertId findDirMaxBruteForce( const Vector3f & dir, const MeshVertPart & mp )
{
    return findDirMaxBruteForce( dir, mp.mesh.points, &mp.mesh.topology.getVertIds( mp.region ) );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region )
{
    return findDirMinMaxBruteForceT( dir, points, region );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector2f & dir, const VertCoords2 & points, const VertBitSet * region )
{
    return findDirMinMaxBruteForceT( dir, points, region );
}

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region )
{
    return findDirMinMaxBruteForce( dir, cloud.points, region ? region : &cloud.validPoints );
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

MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const MeshVertPart & mp )
{
    return findDirMinMaxBruteForce( dir, mp.mesh.points, &mp.mesh.topology.getVertIds( mp.region ) );
}

} //namespace MR
