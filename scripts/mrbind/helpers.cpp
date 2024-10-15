#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRSceneLoad.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPython/MRPython.h"
#include "MRVoxels/MRBoolean.h"

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

    // Detects the format from file extension and loads scene object from it.
    Expected<std::shared_ptr<Object>> loadSceneObject( const std::filesystem::path& path, ProgressCallback callback = {} )
    {
        auto result = SceneLoad::fromAnySupportedFormat( { path }, std::move( callback ) );
        if ( !result.scene || !result.errorSummary.empty() )
            return unexpected( std::move( result.errorSummary ) );

        if ( !result.isSceneConstructed || result.scene->children().size() != 1 )
            return result.scene;
        else
            return result.scene->children().front();
    }

    // saves distance map to a grayscale image file
    //     threshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)
    // minimum (close): 1.0 (white)
    // maximum (far): threshold
    // invalid (infinity): 0.0 (black)
    Expected<void> saveDistanceMapToImage( const DistanceMap& distMap, const std::filesystem::path& filename, float threshold = 1.f / 255 )
    {
        const auto image = convertDistanceMapToImage( distMap, threshold );
        return ImageSave::toAnySupportedFormat( image, filename );
    }

    // load distance map from a grayscale image file
    //     threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid
    Expected<MR::DistanceMap> loadDistanceMapFromImage( const std::filesystem::path& filename, float threshold = 1.f / 255 )
    {
        auto resLoad = ImageLoad::fromAnySupportedFormat( filename );
        if ( !resLoad.has_value() )
            return unexpected( resLoad.error() );
        return convertImageToDistanceMap( *resLoad, threshold );
    }
}

// This stuff makes it so that `MRTest` and our other apps can use the module directly, without having to add it to `PYTHONPATH`.
extern "C" PyObject *PyInit_mrmeshpy();
static MR::PythonFunctionAdder initMrmeshpyModule( "mrmeshpy", &PyInit_mrmeshpy );
