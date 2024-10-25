using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    using VertId = int;


    using OneMeshContours = List<OneMeshContour>;
    using static MR.DotNet.CoordinateConverters;
    using static MR.DotNet.Vector3f;
    using static MR.DotNet.ContoursCut;

    public enum VariantIndex
    {
        Face,
        Edge,
        Vertex
    };
    /// simple point on mesh, represented by primitive id and coordinate in mesh space
    public struct OneMeshIntersection
    {
        public VariantIndex variantIndex;
        public int index;
        public Vector3f coordinate;
    };
    /// one contour on mesh
    public struct OneMeshContour
    {
        public List<OneMeshIntersection> intersections;
        public bool closed;
    };

    
    /// list of contours on mesh


    public class ContoursCut
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshIntersection
        {
            public int primitiveId;
            public byte primitiveIdIndex;
            public MRVector3f coordinate;
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVectorOneMeshIntersection
        {
            public IntPtr data;
            public ulong size;
            public IntPtr reserved;
        };

        // One contour on mesh
        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshContour
        {
            public MRVectorOneMeshIntersection intersections;
            public bool closed;
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVariableEdgeTri
        {
            public MREdgeId edge;
            public MRFaceId tri;
            public bool isEdgeATriB;
        };

       

        /// gets the contours' value at index
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MROneMeshContour mrOneMeshContoursGet( IntPtr contours, ulong index );

        /// gets the contours' size
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrOneMeshContoursSize( IntPtr contours);

        /// deallocates the OneMeshContours object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrOneMeshContoursFree(IntPtr contours);

        // Converts ordered continuous contours of two meshes to OneMeshContours
        // converters is required for better precision in case of degenerations
        // note that contours should not have intersections
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrGetOneMeshIntersectionContours( IntPtr meshA, IntPtr meshB,
                                                                     IntPtr continousContours,
                                                                     bool getMeshAIntersections,
                                                                     ref MRCoordinateConverters converters,
                                                                     IntPtr rigidB2A);
        /// converts ordered continuous contours of two meshes to OneMeshContours
        /// converters is required for better precision in case of degenerations
        /// note that contours should not have intersections
        static public OneMeshContours GetOneMeshIntersectionContours(Mesh meshA, Mesh meshB, ContinousContours contours, bool getMeshAIntersections,
            CoordinateConverters converters, AffineXf3f? rigidB2A = null)
        {
            var mrOneMeshContours = mrGetOneMeshIntersectionContours(meshA.mesh_, meshB.mesh_, contours.mrContours_, getMeshAIntersections, ref converters.conv_, rigidB2A is null ? IntPtr.Zero : rigidB2A.XfAddr());
            int contoursSize = (int)mrOneMeshContoursSize(mrOneMeshContours);
            var oneMeshContours = new OneMeshContours( contoursSize );
            for ( int i = 0; i < contoursSize; i++ )
            {
                var mrOneMeshContour = mrOneMeshContoursGet(mrOneMeshContours, (ulong)i);
                var oneMeshContour = new OneMeshContour
                {
                    intersections = new List<OneMeshIntersection>( (int)mrOneMeshContour.intersections.size ),
                    closed = mrOneMeshContour.closed
                };
                for (int j = 0; j < (int)mrOneMeshContour.intersections.size; j++)
                {
                    var mrOneMeshIntersectionData = mrOneMeshContour.intersections.data;
                    int sizeOfOneMeshIntersection = Marshal.SizeOf(typeof(OneMeshIntersection));
                    var mrOneMeshIntersection = (MROneMeshIntersection)Marshal.PtrToStructure(IntPtr.Add(mrOneMeshIntersectionData, j * sizeOfOneMeshIntersection), typeof(MROneMeshIntersection));
                    var oneMeshIntersection = new OneMeshIntersection();
                    oneMeshIntersection.variantIndex = (VariantIndex)mrOneMeshIntersection.primitiveIdIndex;
                    oneMeshIntersection.index = mrOneMeshIntersection.primitiveId;
                    oneMeshIntersection.coordinate = new Vector3f(mrOneMeshIntersection.coordinate);
                    oneMeshContour.intersections.Add(new OneMeshIntersection
                    {
                        variantIndex = (VariantIndex)mrOneMeshIntersection.primitiveIdIndex,
                        index = mrOneMeshIntersection.primitiveId,
                        coordinate = new Vector3f(mrOneMeshIntersection.coordinate)
                    });
                }
                oneMeshContours.Add(oneMeshContour);
            }
            return oneMeshContours;
        }
    }
}
