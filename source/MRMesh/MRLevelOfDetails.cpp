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
    int numFacesPerObj = totalFaces;
    for ( int d = 0; d < maxDepth; ++d )
        numFacesPerObj /= 2;

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

    std::vector<FaceBitSet> meshParts( spans.size() );
    std::vector<std::shared_ptr<ObjectMesh>> levelObjs( spans.size() );
    ParallelFor( levelObjs, [&]( size_t i )
    {
        auto objMesh = std::make_shared<ObjectMesh>();

        FaceBitSet & region = meshParts[i];
        region.resize( spans[i].end, false );
        region.set( spans[i].beg, spans[i].end - spans[i].beg, true );
        objMesh->setMesh( std::make_shared<Mesh>( mesh.cloneRegion( region ) ) );

        objMesh->setName( std::to_string( maxDepth ) + "_" + std::to_string( i ) );
        levelObjs[i] = std::move( objMesh );
    } );

    DecimateSettings dsettings;
    dsettings.subdivideParts = int( spans.size() );
    dsettings.decimateBetweenParts = false;
    dsettings.partFaces = &meshParts;
    dsettings.maxError = FLT_MAX;
    dsettings.minFacesInPart = numFacesPerObj;
    decimateMesh( mesh, dsettings );

    ParallelFor( levelObjs, [&]( size_t i )
    {
        auto objMesh = std::make_shared<ObjectMesh>();

        const FaceBitSet & region = meshParts[i];
        objMesh->setMesh( std::make_shared<Mesh>( mesh.cloneRegion( region ) ) );

        objMesh->setName( std::to_string( maxDepth-1 ) + "_" + std::to_string( i ) );
        objMesh->addChild( levelObjs[i] );
        levelObjs[i] = std::move( objMesh );
    } );

    auto res = std::make_shared<ObjectMesh>();
    res->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    res->setName( "Level of Details" );
    for ( auto & obj : levelObjs )
        res->addChild( obj );

    return res;
}

} //namespace MR
