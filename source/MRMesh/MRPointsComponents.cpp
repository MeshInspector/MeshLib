#include "MRPointsComponents.h"
#include "MRPointCloud.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRProgressCallback.h"
//#include <algorithm>

namespace MR
{

namespace PointCloudComponents
{

Expected<std::vector<MR::VertBitSet>> getLargestComponents( const PointCloud& pointCloud, float maxDist, int minSize /*= -1*/, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    const auto& validPoints = pointCloud.validPoints;
    if ( minSize < 0 )
        minSize = std::max( int( validPoints.count() ) / 100, 1 );
    ProgressCallback subPc = pc ? subprogress( pc, 0.f, 0.9f ) : pc;
    auto unionStructsRes = PointCloudComponents::getUnionFindStructureVerts( pointCloud, maxDist, nullptr, subPc );
    if ( !unionStructsRes.has_value() )
        return unexpected( unionStructsRes.error() );
    auto& unionStructs = *unionStructsRes;
    auto allRoots = unionStructs.roots();
    HashMap<VertId, int> rootMap;
    std::vector<VertBitSet> components;

    subPc = pc ? subprogress( pc, 0.9f, 1.f ) : pc;
    int counter = 0;
    const float counterMax = float( validPoints.count() );
    const int counterDivider = int( validPoints.count() ) / 100;
    for ( auto v : validPoints )
    {
        VertId root = allRoots[v];
        int componentId = -1;
        if ( rootMap.contains( root ) )
            componentId = rootMap[root];
        else
        {
            componentId = int( components.size() );
            rootMap[root] = componentId;
            components.push_back( VertBitSet( validPoints.find_last() + 1 ) );
        }

        components[componentId].set( v );
        ++counter;
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpected( std::string( "Operation canceled" ) );
    }

    std::vector<VertBitSet> result;
    for ( int i = 0; i < components.size(); ++i )
    {
        if ( components[i].count() > size_t( minSize ) )
            result.push_back( components[i] );
    }

    return result;
}

Expected<UnionFind<MR::VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region /*= nullptr*/, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    VertBitSet vertsRegion = pointCloud.validPoints;
    if ( region )
        vertsRegion &= *region;

    if ( !vertsRegion.any() )
        return {};

    UnionFind<VertId> unionFindStructure( vertsRegion.find_last() + 1 );

    VertId v1;
    int counter = 0;
    const float counterMax = float( vertsRegion.count() );
    const int counterDivider = int( vertsRegion.count() ) / 100;

    for ( auto v0 : vertsRegion )
    {
        findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
            [&] ( VertId v1, const Vector3f& )
        {
            if ( v1.valid() && vertsRegion.test( v1 ) && v1 < v0 )
                unionFindStructure.unite( v0, v1 );
        } );
        ++counter;
        if ( !reportProgress( pc, counter / counterMax, counter, counterDivider ) )
            return unexpected( std::string( "Operation canceled" ) );
    }
    return unionFindStructure;
}

}

}
