#include "MRMeshOverhangs.h"

#include "MRBitSetParallelFor.h"
#include "MRComputeBoundingBox.h"
#include "MRExpandShrink.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

#include <cassert>

namespace MR
{

std::vector<FaceBitSet> findOverhangs( const Mesh& mesh, const FindOverhangsSettings& settings )
{
    MR_TIMER

    assert( std::abs( settings.axis.lengthSq() - 1.f ) < 1e-6f );
    assert( settings.layerHeight > 0.f );
    assert( settings.maxOverhangDistance > 0.f );
    const auto minCos = -settings.layerHeight / std::hypot( settings.layerHeight, settings.maxOverhangDistance );

    const auto isOverhanging = [&] ( FaceId f ) -> bool
    {
        const auto normal = mesh.normal( f );
        const auto cos = dot( settings.axis, normal );
        return cos < minCos;
    };

    // find faces that might create overhangs
    FaceBitSet faces( mesh.topology.lastValidFace() + 1, false );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&] ( FaceId f )
    {
        faces[f] = isOverhanging( f );
    } );

    // simplify the regions...
    expand( mesh.topology, faces, 1 );
    // ...but preserve the overhanging faces
    auto shrunk = faces;
    shrink( mesh.topology, shrunk, 1 );
    BitSetParallelFor( faces - shrunk, [&] ( FaceId f )
    {
        faces[f] = isOverhanging( f );
    } );

    // compute transform from the given axis
    const auto axisXf = AffineXf3f::xfAround( Matrix3f::rotation( Vector3f::plusZ(), settings.axis ), mesh.computeBoundingBox().center() );
    const auto axisMeshBox = computeBoundingBox( mesh.points, nullptr, &axisXf );

    // filter out face regions with too small overhang distance
    auto regions = MeshComponents::getAllComponents( { mesh, &faces } );
    std::vector<FaceBitSet> results;
    results.reserve( regions.size() );
    for ( auto& region : regions )
    {
        const auto vertices = getIncidentVerts( mesh.topology, region );
        const auto axisBox = computeBoundingBox( mesh.points, vertices, &axisXf );
        // don't include the basement region
        if ( axisBox.min.z == axisMeshBox.min.z )
            continue;
        if ( axisBox.size().z > settings.layerHeight )
            results.emplace_back( std::move( region ) );
    }

    return results;
}

} // namespace MR