#include "MRMeshDecimate.h"
#include "MRBitSet.h"
#include "MRMesh.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMesh.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

DecimateResult MeshDecimate::Decimate( Mesh^ mesh, DecimateParameters^ parameters )
{
    if ( !mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    MR::DecimateSettings nativeParams
    {
        .strategy = MR::DecimateStrategy( parameters->strategy ),
        .maxError = parameters->maxError,
        .maxEdgeLen = parameters->maxEdgeLen,
        .maxBdShift = parameters->maxBdShift,
        .maxTriangleAspectRatio = parameters->maxTriangleAspectRatio,
        .criticalTriAspectRatio = parameters->criticalTriAspectRatio,
        .tinyEdgeLength = parameters->tinyEdgeLength,
        .stabilizer = parameters->stabilizer,
        .optimizeVertexPos = parameters->optimizeVertexPos,
        .maxDeletedVertices = parameters->maxDeletedVertices,
        .maxDeletedFaces = parameters->maxDeletedFaces,
        .collapseNearNotFlippable = parameters->collapseNearNotFlippable,
        .touchNearBdEdges = parameters->touchNearBdEdges,
        .touchBdVerts = parameters->touchBdVerts,
        .maxAngleChange = parameters->maxAngleChange,
        .packMesh = parameters->packMesh,
        .subdivideParts = parameters->subdivideParts,
        .decimateBetweenParts = parameters->decimateBetweenParts,
        .minFacesInPart = parameters->minFacesInPart
    };

    MR::FaceBitSet nativeRegion;
    if ( parameters->region )
    {
        nativeRegion = MR::FaceBitSet( parameters->region->bitSet()->m_bits.begin(), parameters->region->bitSet()->m_bits.end() );
        nativeParams.region = &nativeRegion;
    }
    
    auto nativeRes = MR::decimateMesh( *mesh->getMesh(), nativeParams);
    mesh->invalidateCaches();
    if ( parameters->region )
    {
        parameters->region = gcnew FaceBitSet( new MR::BitSet( nativeParams.region->m_bits.begin(), nativeParams.region->m_bits.end() ) );
    }

    DecimateResult res;
    res.errorIntroduced = nativeRes.errorIntroduced;
    res.facesDeleted = nativeRes.facesDeleted;
    res.vertsDeleted = nativeRes.vertsDeleted;

    return res;
}

MR_DOTNET_NAMESPACE_END