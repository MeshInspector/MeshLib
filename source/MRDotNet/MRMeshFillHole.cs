using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;

namespace MR.DotNet
{   
    public enum MultipleEdgesResolveMode
    {
        None = 0, //does not avoid multiple edges
        Simple, //avoids creating edges that already exist in topology
        Strong //makes additional efforts to avoid creating multiple edges
    };

    /** \struct MRFillHoleParams
     * \brief Parameters structure for FillHole\n
     * Structure has some options to control FillHole
     */
    public struct FillHoleParams
    {       
        /** Specifies triangulation metric\n
          * default for FillHole: GetCircumscribedFillMetric\n
          */
        public FillHoleMetric Metric = new FillHoleMetric();

        /// If not null accumulate new faces
        public BitSet? OutNewFaces = null;

        /** If Strong makes additional efforts to avoid creating multiple edges
          *
          * If Simple avoids creating edges that already exist in topology (default)
          *
          * If None does not avoid multiple edges
          */
        public MultipleEdgesResolveMode MultipleEdgesResolveMode = MultipleEdgesResolveMode.Simple;

        /** If true creates degenerate faces band around hole to have sharp angle visualization
          * \warning This flag bad for result topology, most likely you do not need it
          */
        public bool MakeDegenerateBand = false;

        /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
          * must be 2 or larger
          */
        public int MaxPolygonSubdivisions = 20;

        /** Input/output value, if it is present:
          * returns true if triangulation was bad and do not actually fill hole,
          * if triangulation is ok returns false;
          * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
          */
        public bool? StopBeforeBadTriangulation = null;

        public FillHoleParams() {}
    };

    public class MeshFillHole
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFillHoleParams
        {
            public IntPtr metric = IntPtr.Zero;
            public IntPtr outNewFaces = IntPtr.Zero;
            public MultipleEdgesResolveMode multipleEdgesResolveMode = MultipleEdgesResolveMode.Simple;
            public byte makeDegenerateBand = 0;
            public int maxPolygonSubdivisions = 20;
            public IntPtr stopBeforeBadTriangulation = IntPtr.Zero;
            public MRFillHoleParams () {}
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFillHoleParams mrFillHoleParamsNew();


        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFillHole(IntPtr mesh, EdgeId a, ref MRFillHoleParams parameters );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFillHoles(IntPtr mesh, IntPtr pAs, ulong asNum, ref MRFillHoleParams parameters );

        /** \brief Fills hole in mesh\n
          *
          * Fills given hole represented by one of its edges (having no valid left face),
          * default metric: CircumscribedFillMetric
          *
          * \param mesh mesh with hole
          * \param a EdgeId which represents hole (should not have valid left FaceId)
          * \param parameters parameters of hole filling
          *
          */
        unsafe public static void FillHole(ref Mesh mesh, EdgeId a, FillHoleParams parameters)
        {
            MRFillHoleParams mrParam;
            mrParam.metric = parameters.Metric.mrMetric_;
            mrParam.outNewFaces = parameters.OutNewFaces?.bs_ ?? IntPtr.Zero;
            mrParam.multipleEdgesResolveMode = parameters.MultipleEdgesResolveMode;
            mrParam.makeDegenerateBand = parameters.MakeDegenerateBand ? (byte)1 : (byte)0;
            mrParam.maxPolygonSubdivisions = parameters.MaxPolygonSubdivisions;
            
            byte stopBeforeBadTriangulation = 0;
            mrParam.stopBeforeBadTriangulation = parameters.StopBeforeBadTriangulation.HasValue ? new IntPtr( &parameters.StopBeforeBadTriangulation ): IntPtr.Zero;

            mrFillHole(mesh.varMesh(), a, ref mrParam);

            if (parameters.StopBeforeBadTriangulation.HasValue )
            {
                parameters.StopBeforeBadTriangulation = stopBeforeBadTriangulation > 0;
                if ( parameters.StopBeforeBadTriangulation.Value )
                    throw new Exception("Bad triangulation");
            }
        }

        /// fill all holes given by their representative edges in \param edges
        unsafe public static void FillHoles( ref Mesh mesh, List<EdgeId> edges, FillHoleParams parameters )
        {
            MRFillHoleParams mrParam;
            mrParam.metric = parameters.Metric.mrMetric_;
            mrParam.outNewFaces = parameters.OutNewFaces?.bs_ ?? IntPtr.Zero;
            mrParam.multipleEdgesResolveMode = parameters.MultipleEdgesResolveMode;
            mrParam.makeDegenerateBand = parameters.MakeDegenerateBand ? (byte)1 : (byte)0;
            mrParam.maxPolygonSubdivisions = parameters.MaxPolygonSubdivisions;

            byte stopBeforeBadTriangulation = 0;
            mrParam.stopBeforeBadTriangulation = parameters.StopBeforeBadTriangulation.HasValue ? new IntPtr(&stopBeforeBadTriangulation) : IntPtr.Zero;

            int sizeOfEdgeId = Marshal.SizeOf(typeof(EdgeId));
            IntPtr nativeEdges = Marshal.AllocHGlobal(edges.Count * sizeOfEdgeId);

            try
            {                
                for ( int i = 0; i < edges.Count; ++i)
                {
                    Marshal.StructureToPtr(edges[i], IntPtr.Add(nativeEdges, i * sizeOfEdgeId), false);
                }

                mrFillHoles(mesh.varMesh(), nativeEdges, (ulong)edges.Count, ref mrParam);

                if (parameters.StopBeforeBadTriangulation.HasValue)
                {
                    parameters.StopBeforeBadTriangulation = stopBeforeBadTriangulation > 0;
                    if (parameters.StopBeforeBadTriangulation.Value)
                        throw new Exception("Bad triangulation");
                }
            }
            finally
            {
                Marshal.FreeHGlobal(nativeEdges);
            }
        }
    }
}
