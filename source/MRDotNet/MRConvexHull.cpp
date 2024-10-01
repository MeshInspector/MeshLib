#include "MRConvexHull.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRBitSet.h"

#pragma managed( push, off )
#include <MRMesh/MRConvexHull.h>
#include <MRMesh/MRVector.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMesh.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

Mesh^ ConvexHull::MakeConvexHull( VertCoords^ vertCoords, VertBitSet^ validPoints )
{
    MR::VertCoords nativeVertCoords;
    nativeVertCoords.reserve( vertCoords->Count );
    
    for each ( Vector3f^ vert in vertCoords )
        nativeVertCoords.push_back( *vert->vec() );

    MR::VertBitSet nativeValidPoints( validPoints->bitSet()->m_bits.begin(), validPoints->bitSet()->m_bits.end() );

    return gcnew Mesh( new MR::Mesh( std::move( MR::makeConvexHull( nativeVertCoords, nativeValidPoints ) ) ) );
}

Mesh^ ConvexHull::MakeConvexHull( Mesh^ mesh )
{
    return gcnew Mesh( new MR::Mesh( std::move( MR::makeConvexHull( *mesh->getMesh() ) ) ) );
}

Mesh^ ConvexHull::MakeConvexHull( PointCloud^ pointCloud )
{    
    return gcnew Mesh( new MR::Mesh( MR::makeConvexHull( *pointCloud->getPointCloud() ) ) );
}

MR_DOTNET_NAMESPACE_END
