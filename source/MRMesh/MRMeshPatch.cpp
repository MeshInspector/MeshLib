#include "MRMeshPatch.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MREdgePaths.h"
#include "MRMeshComponents.h"
#include "MRBitSet.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet patchMesh( Mesh& mesh, const FaceBitSet& patchBS, const FillHoleNicelySettings& settings /*= {} */ )
{
    MR_TIMER;
    FaceBitSet newFaces;
    auto bounds = delRegionKeepBd( mesh, patchBS );
    auto s = settings;
    for ( const auto& bd : bounds )
    {
        if ( bd.empty() )
            continue;
        auto avgLength = calcPathLength( bd, mesh ) / bd.size();
        if ( settings.subdivideSettings.maxEdgeLen <= 0.0f )
            s.subdivideSettings.maxEdgeLen = float( avgLength ) * 1.5f;
        if ( !mesh.topology.left( bd[0] ) )
            newFaces |= fillHoleNicely( mesh, bd[0], s );
    }
    return newFaces;
}

FaceBitSet patchMeshByGroups( Mesh& mesh, const FaceBitSet& faces, float angleThreshold, const FillHoleNicelySettings& settings )
{
    MR_TIMER;
    // faces of not yet patched groups
    FaceBitSet pendingFaces = faces & mesh.topology.getValidFaces();
    auto groupsMapAndNum = MeshComponents::getAllComponentsMapBySharpEdges( { mesh, &pendingFaces }, angleThreshold );
    auto& groupsMap = groupsMapAndNum.first;
    const int numGroups = groupsMapAndNum.second;
    const auto groups = MeshComponents::getAllComponentsFaces( groupsMap, numGroups, pendingFaces );
    // faces of each group that appeared when patch subdivision split its boundary edges
    Vector<std::vector<FaceId>, RegionId> splitGroupFaces( numGroups );

    FaceBitSet newFaces;
    auto onFaceSplit = [&] ( FaceId oldFace, FaceId newFace )
    {
        if ( contains( pendingFaces, oldFace ) )
        {
            const auto g = groupsMap[oldFace];
            groupsMap.autoResizeSet( newFace, g );
            pendingFaces.autoResizeSet( newFace );
            splitGroupFaces[g].push_back( newFace );
        }
        else if ( contains( newFaces, oldFace ) )
            newFaces.autoResizeSet( newFace );
    };

    auto s = settings;
    s.subdivideSettings.onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
    {
        onFaceSplit( mesh.topology.left( e ), mesh.topology.left( e1 ) );
        onFaceSplit( mesh.topology.right( e ), mesh.topology.right( e1 ) );
        if ( settings.subdivideSettings.onEdgeSplit )
            settings.subdivideSettings.onEdgeSplit( e1, e );
    };

    FaceBitSet group( mesh.topology.faceSize() );
    for ( RegionId r( 0 ); r < RegionId( numGroups ); ++r )
    {
        groups.setComponentBits( r, group );
        for ( auto f : splitGroupFaces[r] )
            group.autoResizeSet( f );
        pendingFaces -= group;
        newFaces |= patchMesh( mesh, group, s );
        group.reset();
    }
    return newFaces;
}

}