#include "MRUniformSampling.h"
#include "MRVector.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRPointsInBall.h"

namespace MR
{

VertBitSet pointUniformSampling( const PointCloud& pointCloud, float distance, ProgressCallback cb )
{
    auto box = pointCloud.getBoundingBox();
    if ( !box.valid() )
        return {};

    auto axis = ( box.max - box.min ).normalized();

    struct VertProj
    {
        float projLength{0};
        VertId id{};
    };

    auto size = pointCloud.validPoints.count();
    if ( size == 0 )
        return {};

    std::vector<VertProj> projes( size );
    int n = 0;
    for ( auto v : pointCloud.validPoints )
    {
        projes[n++] = { dot( pointCloud.points[v],axis ),v };
        // n & 0x7f == 0 ~ n % 128 == 0
        if ( ( ( n & 0x7f ) == 0 ) && cb && !cb( 0.5f * float( n ) / float( size ) ) )
            return {};
    }

    VertBitSet res( pointCloud.validPoints.size() );
    n = 0;
    for ( const auto& proj : projes )
    {
        bool ballHasPrevVert = false;
        findPointsInBall( pointCloud, pointCloud.points[proj.id], distance, [&res,&ballHasPrevVert]( VertId v, const Vector3f& )
        {
            if ( !ballHasPrevVert && res.test( v ) )
                ballHasPrevVert = true;
        } );
        if ( !ballHasPrevVert )
            res.set( proj.id );
        // n & 0x7f == 0 ~ n % 128 == 0
        if ( ( ( ( n++ ) & 0x7f ) == 0 ) && cb && !cb( 0.5f + 0.5f * float( n ) / float( size ) ) )
            return {};
    }

    return res;
}

}
