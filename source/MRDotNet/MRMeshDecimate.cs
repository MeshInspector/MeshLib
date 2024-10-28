using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    /// Defines the order of edge collapses inside Decimate algorithm
    public enum DecimateStrategy
    {
        /// the next edge to collapse will be the one that public introduced minimal error to the surface
        MinimizeError,
        /// the next edge to collapse will be the shortest one
        ShortestEdgeFirst
    };

    public struct DecimateParameters
    {
        public DecimateParameters()
        {}

        public DecimateStrategy strategy = DecimateStrategy.MinimizeError;
        /// for DecimateStrategy::MinimizeError:
        ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
        /// for DecimateStrategy::ShortestEdgeFirst only:
        ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
        public float maxError = 0.001f;
        /// Maximal possible edge length created during decimation
        public float maxEdgeLen = float.MaxValue;
        /// Maximal shift of a boundary during one edge collapse
        public float maxBdShift = float.MaxValue;
        /// Maximal possible aspect ratio of a triangle public introduced during decimation
        public float maxTriangleAspectRatio = 20.0f;
        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public float criticalTriAspectRatio = float.MaxValue;
        /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
        public float tinyEdgeLength = -1;
        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public float stabilizer = 0.001f;
        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public bool optimizeVertexPos = true;
        /// Limit on the number of deleted vertices
        public int maxDeletedVertices = int.MaxValue;
        /// Limit on the number of deleted faces
        public int maxDeletedFaces = int.MaxValue;
        /// Region on mesh to be decimated, it is updated during the operation. If null then whole mesh is decimated
        public BitSet? region = null;
        /// Whether to allow collapse of edges incident to notFlippable edges,
        /// which can move vertices of notFlippable edges unless they are fixed
        public bool collapseNearNotFlippable = false;
        // TODO: edgesToCollapse
        // TODO: twinMap
        /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
        public bool touchNearBdEdges = true;
        /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
        /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
        /// this setting is ignored if touchNearBdEdges=false
        public bool touchBdVerts = true;
        // TODO: bdVerts
        /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
        /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
        public float maxAngleChange = -1;
        /// whether to pack mesh at the end
        public bool packMesh = false;
        /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
        /// unlike \ref mrDecimateParallelMesh it does not create copies of mesh regions, so may take less memory to operate;
        /// IMPORTANT: please call Mesh::PackOptimally before calling decimating with subdivideParts > 1, otherwise performance will be bad
        public int subdivideParts = 1;
        /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
        /// to eliminate small edges near the border of individual parts
        public bool decimateBetweenParts = true;
        // TODO: partFaces
        /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
        public int minFacesInPart = 0;
    };

    public struct DecimateResult
    {
        /// Number deleted verts. Same as the number of performed collapses
        public int vertsDeleted = 0;
        /// Number deleted faces
        public int facesDeleted = 0;
        /// for DecimateStrategy::MinimizeError:
        ///    estimated distance deviation of decimated mesh from the original mesh
        /// for DecimateStrategy::ShortestEdgeFirst:
        ///    the shortest remaining edge in the mesh
        public float errorIntroduced = 0;
        public DecimateResult() {}
    };
    public struct MRDecimateParameters
    {
        public DecimateStrategy strategy = DecimateStrategy.MinimizeError;
        public float maxError = 0.001f;
        public float maxEdgeLen = float.MaxValue;
        public float maxBdShift = float.MaxValue;
        public float maxTriangleAspectRatio = 20.0f;
        public float criticalTriAspectRatio = float.MaxValue;
        public float tinyEdgeLength = -1;
        public float stabilizer = 0.001f;
        public bool optimizeVertexPos = true;
        public int maxDeletedVertices = int.MaxValue;
        public int maxDeletedFaces = int.MaxValue;
        public IntPtr region = IntPtr.Zero;
        public bool collapseNearNotFlippable = false;
        public bool touchNearBdEdges = true;
        public bool touchBdVerts = true;
        public float maxAngleChange = -1;
        public bool packMesh = false;
        public IntPtr progressCallback = IntPtr.Zero;
        public int subdivideParts = 1;
        public bool decimateBetweenParts = true;
        public int minFacesInPart = 0;
        public MRDecimateParameters() {}
    };

    public class MeshDecimate
    {

        /// Collapse edges in mesh region according to the settings
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern DecimateResult mrDecimateMesh( IntPtr mesh, ref MRDecimateParameters settings );

        public static DecimateResult Decimate( Mesh mesh, DecimateParameters settings )
        {
            MRDecimateParameters mrParameters = new MRDecimateParameters();
            mrParameters.strategy = settings.strategy;
            mrParameters.maxError = settings.maxError;
            mrParameters.maxEdgeLen = settings.maxEdgeLen;
            mrParameters.maxBdShift = settings.maxBdShift;
            mrParameters.maxTriangleAspectRatio = settings.maxTriangleAspectRatio;
            mrParameters.criticalTriAspectRatio = settings.criticalTriAspectRatio;
            mrParameters.tinyEdgeLength = settings.tinyEdgeLength;
            mrParameters.stabilizer = settings.stabilizer;
            mrParameters.optimizeVertexPos = settings.optimizeVertexPos;
            mrParameters.maxDeletedVertices = settings.maxDeletedVertices;
            mrParameters.maxDeletedFaces = settings.maxDeletedFaces;
            mrParameters.region = settings.region is null ? (IntPtr)null : settings.region.bs_;
            mrParameters.collapseNearNotFlippable = settings.collapseNearNotFlippable;
            mrParameters.touchNearBdEdges = settings.touchNearBdEdges;
            mrParameters.touchBdVerts = settings.touchBdVerts;
            mrParameters.maxAngleChange = settings.maxAngleChange;
            mrParameters.packMesh = settings.packMesh;
            mrParameters.subdivideParts = settings.subdivideParts;
            mrParameters.decimateBetweenParts = settings.decimateBetweenParts;
            mrParameters.progressCallback = IntPtr.Zero;
            mrParameters.minFacesInPart = settings.minFacesInPart;

            return mrDecimateMesh( mesh.mesh_, ref mrParameters);
        }

    }
}
