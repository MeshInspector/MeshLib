#include "MRCoordinateConverters.h"
#include "MRBitSet.h"


#pragma managed( push, off )
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMeshCollidePrecise.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

CoordinateConverters::CoordinateConverters( MeshPart meshA, MeshPart meshB )
{
    MR::FaceBitSet nativeRegionA;
    MR::FaceBitSet nativeRegionB;
    if ( meshA.region )
        nativeRegionA = MR::FaceBitSet( meshA.region->bitSet()->m_bits.begin(), meshA.region->bitSet()->m_bits.end() );
    if ( meshB.region )
        nativeRegionB = MR::FaceBitSet( meshB.region->bitSet()->m_bits.begin(), meshB.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMeshA( *meshA.mesh->getMesh(), meshA.region ? &nativeRegionA : nullptr );
    MR::MeshPart nativeMeshB( *meshB.mesh->getMesh(), meshB.region ? &nativeRegionB : nullptr );

    auto nativeConverters = MR::getVectorConverters( nativeMeshA, nativeMeshB );
    convertToFloatVector_ = new MR::ConvertToFloatVector( nativeConverters.toFloat );
    convertToIntVector_ = new MR::ConvertToIntVector( nativeConverters.toInt );
}

CoordinateConverters::~CoordinateConverters()
{
    delete convertToFloatVector_;
    delete convertToIntVector_;
}

MR::CoordinateConverters CoordinateConverters::ToNative()
{
    return MR::CoordinateConverters { .toInt = *convertToIntVector_, .toFloat = *convertToFloatVector_ };
}

MR_DOTNET_NAMESPACE_END