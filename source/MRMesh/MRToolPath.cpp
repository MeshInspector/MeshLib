#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MRToolPath.h"
#include "MRSurfacePath.h"
#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRMesh/MRExtractIsolines.h"

namespace MR
{
std::shared_ptr<Polyline3> getToolPath( Mesh& mesh, float millRadius, float voxelSize, float sectionsStep, float critLength )
{
    const Vector3f normal = Vector3f::minusZ();
    FixUndercuts::fixUndercuts( mesh, normal, voxelSize );

    OffsetParameters offsetParams;
    offsetParams.voxelSize = voxelSize;

    auto resMesh = offsetMesh( mesh, millRadius, offsetParams );
    if ( !resMesh.has_value() )
        return {};

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.min.z - millRadius;

    const float dragSpeed = millRadius * 0.001f;
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.min - normal * dragSpeed );
    const int steps = int( std::floor( ( plane.d + box.max.z ) / sectionsStep ) );

    Contour3f toolPath{ { 0, 0, safeZ } };

    MeshEdgePoint prevEdgePoint;

    const float critLengthSq = critLength * critLength;
    for ( int step = 0; step < steps ; ++step )
    {        
        for ( const auto& section : extractPlaneSections( mesh, Plane3f{ plane.n, plane.d - sectionsStep * step } ) )
        {
            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours().front();
            auto nextEdgePointIt = section.begin();

            float minDistSq = FLT_MAX;

            if ( prevEdgePoint.e.valid() )
            {
                for ( auto it = section.begin(); it < section.end(); ++it )
                {
                    float distSq = ( mesh.edgePoint( *it ) - mesh.edgePoint( prevEdgePoint ) ).lengthSq();
                    if ( distSq < minDistSq )
                    {
                        minDistSq = distSq;
                        nextEdgePointIt = it;
                    }
                }
            }

            const auto pivotIt = contours.begin() + std::distance( section.begin(), nextEdgePointIt );
            
            if ( !prevEdgePoint.e.valid() || minDistSq > critLengthSq )
            {
                toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                toolPath.push_back( { pivotIt->x, pivotIt->y, safeZ } );
            }
            else
            {
                Polyline3 transit;
                const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt );
                if ( sp.has_value() && sp->size() > 1 )
                {
                    transit.addFromSurfacePath( mesh, *sp );
                    const auto transitContours = transit.contours().front();
                    toolPath.insert( toolPath.end(), transitContours.begin(), transitContours.end() );
                }
            }
            
            toolPath.insert( toolPath.end(),  pivotIt, contours.end() );
            toolPath.insert( toolPath.end(), contours.begin(), pivotIt + 1 );

            prevEdgePoint = *nextEdgePointIt;
        }        
    }

    return std::make_shared<Polyline3>( Contours3f{ toolPath } );
}
}
#endif
