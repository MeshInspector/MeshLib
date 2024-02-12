#include "MRPointsComponents.h"
#include "MRPointCloud.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
//#include <algorithm>

namespace MR
{

namespace PointCloudComponents
{

std::vector<MR::VertBitSet> getBigComponents( const PointCloud& pointCloud, float maxDist, int minSize /*= -1*/ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    const auto& validPoints = pointCloud.validPoints;
    if ( minSize < 0 )
        minSize = std::max( int( validPoints.count() ) / 100, 1 );
    auto unionSstructs = PointCloudComponents::getUnionFindStructureVerts( pointCloud, maxDist );
    auto allRoots = unionSstructs.roots();
    std::vector<VertBitSet> components;
    std::vector<VertId> componentRoots;

    for ( auto v : validPoints )
    {
        auto componentId = -1;
        VertId root = allRoots[v];
        for ( int i = 0; i < componentRoots.size(); ++i )
            if ( componentRoots[i] == root )
            {
                componentId = i;
                break;
            }

        if ( componentId == -1 )
        {
            componentRoots.push_back( root );
            components.push_back( VertBitSet( validPoints.find_last() + 1 ) );
            componentId = int( componentRoots.size() ) - 1;
        }

        components[componentId].set( v );
    }

    std::vector<VertBitSet> result;
    for ( int i = 0; i < components.size(); ++i )
    {
        if ( components[i].count() > size_t( minSize ) )
            result.push_back( components[i] );
    }

    return result;
}

MR::UnionFind<MR::VertId> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region /*= nullptr*/ )
{
    MR_TIMER

    VertBitSet vertsRegion = pointCloud.validPoints;
    if ( region )
        vertsRegion &= *region;

    if ( !vertsRegion.any() )
        return {};

    UnionFind<VertId> unionFindStructure( vertsRegion.find_last() + 1 );

    VertId v1;
    for ( auto v0 : vertsRegion )
    {
        findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
            [&] ( VertId v1, const Vector3f& )
        {
            if ( v1.valid() && vertsRegion.test( v1 ) && v1 < v0 )
                unionFindStructure.unite( v0, v1 );
        } );
    }
    return unionFindStructure;
}

}

}
