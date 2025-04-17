#include "MRMeshFixer.h"
#include "MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "detail/TypeCast.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshPart )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST2( FixMeshDegeneraciesParams::Mode, MRFixMeshDegeneraciesParamsMode )
REGISTER_AUTO_CAST2( std::string, MRString )

#define COPY_FROM( obj, field ) . field = auto_cast( ( obj ). field ),

MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( findHoleComplicatingFaces( mesh ) );
}

MRFaceBitSet* mrFindDegenerateFaces( const MRMeshPart* mp_, float criticalAspectRatio, MRProgressCallback cb_, MRString** errorString )
{
    ARG( mp );
    auto cb = [cb_]( float progress ) -> bool { return cb_( progress ); };
    auto res = cb_ ? findDegenerateFaces( mp, criticalAspectRatio, cb ) : findDegenerateFaces( mp, criticalAspectRatio );
    if ( res )
        RETURN_NEW( *res );

    if ( errorString && !res )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;    
}

MRUndirectedEdgeBitSet* mrFindShortEdges( const MRMeshPart* mp_, float criticalLength, MRProgressCallback cb_, MRString** errorString )
{
    ARG( mp );
    auto cb = [cb_]( float progress ) -> bool { return cb_( progress ); };
    auto res = cb_ ? findShortEdges( mp, criticalLength, cb ) : findShortEdges( mp, criticalLength );
    if ( res )
        RETURN_NEW( *res );

    if ( errorString && !res )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}

void fixMultipleEdges( MRMesh* mesh_, const MRMultipleEdge* multipleEdges, size_t multipleEdgesNum )
{
    ARG( mesh );
    std::vector<MultipleEdge> multipleEdgesVec( (MultipleEdge*)multipleEdges, ( MultipleEdge* )multipleEdges + multipleEdgesNum );
    fixMultipleEdges( mesh, multipleEdgesVec );
}

void findAndFixMultipleEdges( MRMesh* mesh_ )
{
    ARG( mesh );
    fixMultipleEdges( mesh );
}

MRFixMeshDegeneraciesParams mrFixMeshDegeneraciesParamsNew( void )
{
    static const FixMeshDegeneraciesParams def {};
    return {
        COPY_FROM( def, maxDeviation )
        COPY_FROM( def, tinyEdgeLength )
        COPY_FROM( def, criticalTriAspectRatio )
        COPY_FROM( def, maxAngleChange )
        COPY_FROM( def, stabilizer )
        .region = NULL,
        COPY_FROM( def, mode )
        .cb = NULL,
    };
}

void mrFixMeshDegeneracies( MRMesh* mesh_, const MRFixMeshDegeneraciesParams* params_, MRString** errorString )
{
    ARG( mesh );

    FixMeshDegeneraciesParams params;
    if ( params_ )
    {
        const auto& src = *params_;
        params = FixMeshDegeneraciesParams {
            COPY_FROM( src, maxDeviation )
            COPY_FROM( src, tinyEdgeLength )
            COPY_FROM( src, criticalTriAspectRatio )
            COPY_FROM( src, maxAngleChange )
            COPY_FROM( src, stabilizer )
            COPY_FROM( src, region )
            COPY_FROM( src, mode )
            .cb = src.cb,
        };
    }

    auto res = fixMeshDegeneracies( mesh, params );
    if ( res )
        return;

    if ( errorString )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );
}
