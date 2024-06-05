#include "MRLevelOfDetails.h"
#include "MRObjectMesh.h"
#include "MRMesh.h"
#include "MRMeshDecimate.h"
#include "MRBuffer.h"
#include "MRTimer.h"
#include "MRParallelFor.h"

namespace MR
{

std::shared_ptr<ObjectMesh> makeLevelOfDetails( Mesh && mesh, int maxDepth )
{
    MR_TIMER
    assert( maxDepth > 0 );

    mesh.packOptimally( false );

    const int totalFaces = mesh.topology.numValidFaces();
    //int numFacesPerObj = totalFaces;
    //for ( int d = 0; d < maxDepth; ++d )
    //    numFacesPerObj /= 2;

    struct FaceSpan
    {
        FaceId beg, end;
    };

    std::vector<FaceSpan> spans;
    spans.push_back( { 0_f, FaceId( totalFaces ) } );
    for ( int d = 1; d < maxDepth; ++d )
    {
        std::vector<FaceSpan> nextSpans;
        for ( const auto & span : spans )
        {
            auto mid = span.beg + ( span.end - span.beg ) / 2;
            nextSpans.push_back( { span.beg, mid } );
            nextSpans.push_back( { mid, span.end } );
        }
        spans = std::move( nextSpans );
    }

    std::vector<std::shared_ptr<ObjectMesh>> level( spans.size() );
    ParallelFor( level, [&]( size_t i )
    {
        auto objMesh = std::make_shared<ObjectMesh>();

        FaceBitSet region;
        region.resize( spans[i].end, false );
        region.set( spans[i].beg, spans[i].end - spans[i].beg, true );
        objMesh->setMesh( std::make_shared<Mesh>( mesh.cloneRegion( region ) ) );

        //auto pMesh = std::make_shared<Mesh>();
        //pMesh->addPartBy( mesh, spans[i].beg, spans[i].end, spans[i].end - spans[i].beg );
        //objMesh->setMesh( std::move( pMesh ) );

        objMesh->setName( std::to_string( maxDepth ) + "_" + std::to_string( i ) );
        level[i] = std::move( objMesh );
    } );

    auto res = std::make_shared<ObjectMesh>();
    res->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    res->setName( "Level of Details" );
    for ( auto & obj : level )
        res->addChild( obj );

    return res;
}

} //namespace MR
