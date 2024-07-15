#include "MRMeshMetrics.h"

#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRId.h"

using namespace MR;

MRFillHoleMetric* mrFillHoleMetricNew( MRFillTriangleMetric triangleMetric, MRFillEdgeMetric edgeMetric, MRFillCombineMetric combineMetric )
{
    FillHoleMetric res {
        .triangleMetric = [triangleMetric] ( VertId v1_, VertId v2_, VertId v3_ ) -> double
        {
            const auto& v1 = reinterpret_cast<MRVertId&>( v1_ );
            const auto& v2 = reinterpret_cast<MRVertId&>( v2_ );
            const auto& v3 = reinterpret_cast<MRVertId&>( v3_ );
            return triangleMetric( v1, v2, v3 );
        },
        .edgeMetric = [edgeMetric] ( VertId v1_, VertId v2_, VertId v3_, VertId v4_ ) -> double
        {
            const auto& v1 = reinterpret_cast<MRVertId&>( v1_ );
            const auto& v2 = reinterpret_cast<MRVertId&>( v2_ );
            const auto& v3 = reinterpret_cast<MRVertId&>( v3_ );
            const auto& v4 = reinterpret_cast<MRVertId&>( v4_ );
            return edgeMetric( v1, v2, v3, v4 );
        },
        .combineMetric = combineMetric,
    };
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

void mrFillHoleMetricFree( MRFillHoleMetric* metric )
{
    delete reinterpret_cast<FillHoleMetric*>( metric );
}

double mrCalcCombinedFillMetric( const MRMesh* mesh_, const MRFaceBitSet* filledRegion_, const MRFillHoleMetric* metric_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    const auto& filledRegion = *reinterpret_cast<const FaceBitSet*>( filledRegion_ );
    const auto& metric = *reinterpret_cast<const FillHoleMetric*>( metric_ );

    return calcCombinedFillMetric( mesh, filledRegion, metric );
}

MRFillHoleMetric* mrGetCircumscribedMetric( const MRMesh* mesh_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );

    auto res = getCircumscribedMetric( mesh );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

MRFillHoleMetric* mrGetPlaneFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    const auto& e = reinterpret_cast<const EdgeId&>( e_ );

    auto res = getPlaneFillMetric( mesh, e );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

MRFillHoleMetric* mrGetPlaneNormalizedFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    const auto& e = reinterpret_cast<const EdgeId&>( e_ );

    auto res = getPlaneNormalizedFillMetric( mesh, e );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

MRFillHoleMetric* mrGetComplexFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    const auto& e = reinterpret_cast<const EdgeId&>( e_ );

    auto res = getComplexFillMetric( mesh, e );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

MRFillHoleMetric* mrGetUniversalMetric( const MRMesh* mesh_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );

    auto res = getUniversalMetric( mesh );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}

MRFillHoleMetric* mrGetMinAreaMetric( const MRMesh* mesh_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );

    auto res = getMinAreaMetric( mesh );
    return reinterpret_cast<MRFillHoleMetric*>( new FillHoleMetric( std::move( res ) ) );
}
