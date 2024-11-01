using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MR.DotNet
{
    using OneMeshContours = List<OneMeshContour>;
    using static MR.DotNet.CoordinateConverters;
    using static MR.DotNet.Vector3f;
    /// represents primitive type
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

        public OneMeshIntersection(VariantIndex variantIndex, int index, Vector3f coordinate) 
        { 
            this.variantIndex = variantIndex; 
            this.index = index; 
            this.coordinate = coordinate; 
        }
    };
    /// one contour on mesh
    public struct OneMeshContour
    {
        public List<OneMeshIntersection> intersections;
        public bool closed;

        public OneMeshContour(List<OneMeshIntersection> intersections, bool closed)
        {
            this.intersections = intersections;
            this.closed = closed;
        }
    };

    /// list of contours on mesh
    public class ContoursCut
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshIntersection
        {
            public int primitiveId = 0;
            public byte primitiveIdIndex = 0;
            public MRVector3f coordinate;
            public MROneMeshIntersection() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshContour
        {
            public IntPtr data = IntPtr.Zero;
            public ulong size = 0;
            public IntPtr reserved = IntPtr.Zero;
            [MarshalAs(UnmanagedType.U1)]
            public bool closed = false;
            public MROneMeshContour() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVariableEdgeTri
        {
            public EdgeId edge = new EdgeId();
            public FaceId tri = new FaceId();
            [MarshalAs(UnmanagedType.U1)]
            public bool isEdgeATriB = false;
            public MRVariableEdgeTri() { }
        };       

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrOneMeshContoursGet( IntPtr contours, ulong index, out MROneMeshContour outContour );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrOneMeshContoursSize( IntPtr contours);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrOneMeshContoursFree(IntPtr contours);

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
                MROneMeshContour mrOneMeshContour;
                mrOneMeshContoursGet(mrOneMeshContours, (ulong)i, out mrOneMeshContour);
                var oneMeshContour = new OneMeshContour
                {
                    intersections = new List<OneMeshIntersection>( (int)mrOneMeshContour.size ),
                    closed = mrOneMeshContour.closed
                };
                for (int j = 0; j < (int)mrOneMeshContour.size; j++)
                {
                    var mrOneMeshIntersectionData = mrOneMeshContour.data;
                    int sizeOfOneMeshIntersection = Marshal.SizeOf(typeof(MROneMeshIntersection));
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
