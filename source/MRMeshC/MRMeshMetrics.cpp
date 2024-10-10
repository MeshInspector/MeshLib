#include "MRMeshMetrics.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRId.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( FillHoleMetric )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( VertId )

MRFillHoleMetric* mrFillHoleMetricNew( MRFillTriangleMetric triangleMetric, MRFillEdgeMetric edgeMetric, MRFillCombineMetric combineMetric )
{
    RETURN_NEW( FillHoleMetric {
        .triangleMetric = [triangleMetric] ( VertId v1_, VertId v2_, VertId v3_ ) -> double
        {
            ARG_VAL( v1 ); ARG_VAL( v2 ); ARG_VAL( v3 );
            return triangleMetric( v1, v2, v3 );
        },
        .edgeMetric = [edgeMetric] ( VertId v1_, VertId v2_, VertId v3_, VertId v4_ ) -> double
        {
            ARG_VAL( v1 ); ARG_VAL( v2 ); ARG_VAL( v3 ); ARG_VAL( v4 );
            return edgeMetric( v1, v2, v3, v4 );
        },
        .combineMetric = combineMetric,
    } );
}

void mrFillHoleMetricFree( MRFillHoleMetric* metric_ )
{
    ARG_PTR( metric );
    delete metric;
}

double mrCalcCombinedFillMetric( const MRMesh* mesh_, const MRFaceBitSet* filledRegion_, const MRFillHoleMetric* metric_ )
{
    ARG( mesh ); ARG( filledRegion ); ARG( metric );
    return calcCombinedFillMetric( mesh, filledRegion, metric );
}

MRFillHoleMetric* mrGetCircumscribedMetric( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( getCircumscribedMetric( mesh ) );
}

MRFillHoleMetric* mrGetPlaneFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    ARG( mesh ); ARG_VAL( e );
    RETURN_NEW( getPlaneFillMetric( mesh, e ) );
}

MRFillHoleMetric* mrGetPlaneNormalizedFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    ARG( mesh ); ARG_VAL( e );
    RETURN_NEW( getPlaneNormalizedFillMetric( mesh, e ) );
}

MRFillHoleMetric* mrGetComplexFillMetric( const MRMesh* mesh_, MREdgeId e_ )
{
    ARG( mesh ); ARG_VAL( e );
    RETURN_NEW( getComplexFillMetric( mesh, e ) );
}

MRFillHoleMetric* mrGetUniversalMetric( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( getUniversalMetric( mesh ) );
}

MRFillHoleMetric* mrGetMinAreaMetric( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( getMinAreaMetric( mesh ) );
}
