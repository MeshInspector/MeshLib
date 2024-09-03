#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRBoolean.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointsToMeshProjector.h"

// Only the functions that should be exported should be in `MR::Extra`. Place everything else somewhere outside.
// Note that the comments are pasted to Python too.

namespace MR::Extra
{
    // Fix self-intersections by converting to voxels and back.
    void fixSelfIntersections( Mesh& mesh, float voxelSize )
    {
        MeshVoxelsConverter convert;
        convert.voxelSize = voxelSize;
        auto gridA = convert(mesh);
        mesh = convert(gridA);
    }

    // Subtract mesh B from mesh A.
    Mesh voxelBooleanSubtract( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
    {
        MeshVoxelsConverter convert;
        convert.voxelSize = voxelSize;
        auto gridA = convert(mesh1);
        auto gridB = convert(mesh2);
        gridA -= gridB;
        return convert(gridA);
    }

    // Unite mesh A and mesh B.
    Mesh voxelBooleanUnite( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
    {
        MeshVoxelsConverter convert;
        convert.voxelSize = voxelSize;
        auto gridA = convert(mesh1);
        auto gridB = convert(mesh2);
        gridA += gridB;
        return convert( gridA );
    }

    // Intersect mesh A and mesh B.
    Mesh voxelBooleanIntersect( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
    {
        MeshVoxelsConverter convert;
        convert.voxelSize = voxelSize;
        auto gridA = convert(mesh1);
        auto gridB = convert(mesh2);
        gridA *= gridB;
        return convert( gridA );
    }

    // Computes signed distances from all mesh points to refMesh.
    // `refMesh` - all points will me projected to this mesh
    // `mesh` - this mesh points will be projected
    // `refXf` - world transform for refMesh
    // `upDistLimitSq` - upper limit on the distance in question, if the real distance is larger than the returning upDistLimit
    // `loDistLimitSq` - low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    VertScalars projectAllMeshVertices( const Mesh& refMesh, const Mesh& mesh, const AffineXf3f* refXf = nullptr, const AffineXf3f* xf = nullptr, float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0.0f )
    {
        PointsToMeshProjector projector;
        projector.updateMeshData( &refMesh );
        std::vector<MeshProjectionResult> mpRes( mesh.points.vec_.size() );
        projector.findProjections( mpRes, mesh.points.vec_, xf, refXf, upDistLimitSq, loDistLimitSq );
        VertScalars res( mesh.topology.lastValidVert() + 1, std::sqrt( upDistLimitSq ) );

        AffineXf3f fullXf;
        if ( refXf )
            fullXf = refXf->inverse();
        if ( xf )
            fullXf = fullXf * ( *xf );

        BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
        {
            const auto& mpResV = mpRes[v.get()];
            auto& resV = res[v];

            resV = mpResV.distSq;
            if ( mpResV.mtp.e )
                resV = refMesh.signedDistance( fullXf( mesh.points[v] ), mpResV );
            else
                resV = std::sqrt( resV );
        } );
        return res;
    }

    // Merge a list of meshes to one mesh.
    Mesh mergeMeshes( const std::vector<std::shared_ptr<MR::Mesh>>& meshes )
    {
        Mesh res;
        for ( const auto& m : meshes )
            res.addPart( *m );
        return res;
    }

    // Return faces with at least one edge longer than the specified length.
    FaceBitSet getFacesByMinEdgeLength( const Mesh& mesh, float minLength )
    {
        using namespace MR;
        FaceBitSet resultFaces( mesh.topology.getValidFaces().size() );
        float minLengthSq = minLength * minLength;
        for ( auto ue : undirectedEdges( mesh.topology ) )
        {
            if ( mesh.edgeLengthSq( ue ) > minLengthSq )
            {
                auto l = mesh.topology.left( ue );
                auto r = mesh.topology.right( ue );
                if ( l )
                    resultFaces.set( l );
                if ( r )
                    resultFaces.set( r );
            }
        }
        return resultFaces;
    }
}
