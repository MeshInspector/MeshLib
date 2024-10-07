#include "MRConvexHull.h"

#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MRMesh.h"

using namespace MR;

MRMesh* mrMakeConvexHullFromMesh( const MRMesh* mesh_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );

    auto result = makeConvexHull( mesh );

    return reinterpret_cast<MRMesh*>( new Mesh( std::move( result ) ) );
}

MRMesh* mrMakeConvexHullFromPointCloud( const MRPointCloud* pointCloud_ )
{
    const auto& pointCloud = *reinterpret_cast<const PointCloud*>( pointCloud_ );

    auto result = makeConvexHull( pointCloud );

    return reinterpret_cast<MRMesh*>( new Mesh( std::move( result ) ) );
}
